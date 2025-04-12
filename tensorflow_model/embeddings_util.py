import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import librosa
import json
import glob
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Define the function to load and preprocess audio if not already imported
def load_and_preprocess_audio_file(audio_path, sample_rate=16000, n_mels=40, n_fft=1024, hop_length=512, max_duration=None, add_batch_dim=False):
    """
    Load and preprocess a single audio file for training or inference.
    Using the same function for both training and inference ensures consistency.
    """
    # Load audio file
    audio, _ = librosa.load(audio_path, sr=sample_rate, res_type='kaiser_fast')
    
    # Trim or pad if max_duration is specified
    if max_duration is not None:
        max_samples = int(max_duration * sample_rate)
        if len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        else:
            audio = audio[:max_samples]
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to log scale (dB)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    
    # Transpose to have time as first dimension
    mel_spec = mel_spec.T
    
    # Add batch dimension if requested (for inference)
    if add_batch_dim:
        mel_spec = np.expand_dims(mel_spec, axis=0)
    
    return mel_spec

def load_model_and_metadata(model_path, metadata_path=None):
    """
    Load the trained model and its metadata
    """
    # Import CRNN locally to avoid circular imports
    from Models import CRNN
    
    # If metadata path is not provided, try to infer it
    if metadata_path is None:
        metadata_path = os.path.join(os.path.dirname(model_path), 'model_metadata.json')
    
    # Load metadata if it exists
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        max_duration = metadata.get('max_duration')
        sample_rate = metadata.get('sample_rate', 16000)
        n_mels = metadata.get('n_mels', 40)
        n_fft = metadata.get('n_fft', 1024)
        hop_length = metadata.get('hop_length', 512)
        classes = metadata.get('classes', [])
        
        print(f"Loaded metadata: SR={sample_rate}, Mel bins={n_mels}, Max duration={max_duration}")
    else:
        # Use default values if metadata doesn't exist
        print("No metadata found, using default values")
        max_duration = 5.63  # Use the value from your example
        sample_rate = 16000
        n_mels = 40
        n_fft = 1024
        hop_length = 512
        
        # Try to load classes from classes.txt
        if os.path.exists('./dataset_cache/classes.txt'):
            with open('./dataset_cache/classes.txt', 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
        else:
            classes = []
    
    # Load the model
    try:
        # First try loading directly
        model = keras.models.load_model(model_path, custom_objects={'CRNN': CRNN})
    except:
        # If that fails, try initializing the model and loading weights
        num_classes = len(classes) if classes else 74
        model = CRNN(num_classes=num_classes)
        
        # Calculate the max length in frames from the max duration
        max_length = int(np.ceil(max_duration * sample_rate / hop_length)) if max_duration else 100
        
        # Build the model
        model.build(input_shape=(None, max_length, n_mels))
        
        # Load weights
        try:
            model.load_weights(model_path)
            print("Successfully loaded model weights")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    
    return model, classes, max_duration, sample_rate, n_mels, n_fft, hop_length

def create_embedding_model(model):
    """
    Create a new model that outputs embeddings from the main model
    """
    # Find the layer before the final dense layer (typically the second-to-last layer)
    embedding_layer_name = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense) and layer.name != model.layers[-1].name:
            embedding_layer_name = layer.name
            break
    
    if embedding_layer_name is None:
        # If no dense layer found except the final one, use the last non-dense layer
        for i in range(len(model.layers)-2, -1, -1):
            if not isinstance(model.layers[i], keras.layers.Dense):
                embedding_layer_name = model.layers[i].name
                break
    
    # If still not found, use the layer before final layer
    if embedding_layer_name is None:
        embedding_layer_name = model.layers[-2].name
    
    print(f"Using {embedding_layer_name} as embedding layer")
    
    # Create a new model that outputs embeddings
    embedding_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer(embedding_layer_name).output
    )
    
    return embedding_model

def extract_audio_embedding(model, audio_path, sample_rate=16000, n_mels=40, n_fft=1024, hop_length=512, max_duration=None):
    """
    Extract embedding from an audio file
    """
    # Load and preprocess audio
    mel_spec = load_and_preprocess_audio_file(
        audio_path, 
        sample_rate=sample_rate, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        max_duration=max_duration,
        add_batch_dim=True
    )
    
    # Get embedding
    embedding = model.predict(mel_spec, verbose=0)
    
    return embedding

def calculate_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings
    """
    # Reshape if needed
    if len(embedding1.shape) > 2:
        embedding1 = embedding1.reshape(embedding1.shape[0], -1)
    if len(embedding2.shape) > 2:
        embedding2 = embedding2.reshape(embedding2.shape[0], -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    
    return similarity

def find_similar_audios(reference_embedding, audio_directory, embedding_model, top_n=5, **kwargs):
    """
    Find similar audios to a reference audio based on embedding similarity
    """
    # Get all audio files in the directory
    audio_files = []
    for root, dirs, files in os.walk(audio_directory):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.flac'):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print("No audio files found in the directory")
        return []
    
    # Calculate similarities
    similarities = []
    for file_path in audio_files:
        try:
            # Extract embedding
            embedding = extract_audio_embedding(embedding_model, file_path, **kwargs)
            
            # Calculate similarity
            similarity = calculate_similarity(reference_embedding, embedding)
            
            # Store result
            similarities.append({
                'file': file_path,
                'similarity': similarity,
                'class': os.path.basename(os.path.dirname(file_path))
            })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Return top N results
    return similarities[:top_n]

def visualize_embedding_clusters(embedding_model, audio_directory, classes=None, n_samples_per_class=5, **kwargs):
    """
    Visualize clusters of audio embeddings using t-SNE
    """
    from sklearn.manifold import TSNE
    
    # Get sample audio files from each class
    audio_files = []
    labels = []
    
    # If classes is not provided, discover all classes from the directory structure
    if classes is None:
        classes = [d for d in os.listdir(audio_directory) if os.path.isdir(os.path.join(audio_directory, d))]
    
    # Collect audio files from each class
    for class_name in classes:
        class_path = os.path.join(audio_directory, class_name)
        if not os.path.isdir(class_path):
            continue
            
        # Get all audio files in the class directory
        class_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            class_files.extend(glob.glob(os.path.join(class_path, ext)))
        
        # Select random samples
        if len(class_files) > n_samples_per_class:
            import random
            selected_files = random.sample(class_files, n_samples_per_class)
        else:
            selected_files = class_files
        
        audio_files.extend(selected_files)
        labels.extend([class_name] * len(selected_files))
    
    # Extract embeddings for all audio files
    embeddings = []
    final_labels = []
    final_files = []
    
    print(f"Extracting embeddings for {len(audio_files)} audio files...")
    for i, (file_path, label) in enumerate(zip(audio_files, labels)):
        try:
            if i % 10 == 0:
                print(f"Processing file {i+1}/{len(audio_files)}")
                
            # Extract embedding
            embedding = extract_audio_embedding(embedding_model, file_path, **kwargs)
            
            # Flatten embedding if needed
            if len(embedding.shape) > 2:
                embedding = embedding.reshape(embedding.shape[0], -1)
                
            embeddings.append(embedding[0])
            final_labels.append(label)
            final_files.append(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # Apply t-SNE for dimensionality reduction
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot clusters
    plt.figure(figsize=(12, 10))
    
    # Get unique labels and assign colors
    unique_labels = list(set(final_labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each class
    for label, color in zip(unique_labels, colors):
        mask = [l == label for l in final_labels]
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            color=color,
            label=label,
            alpha=0.7
        )
    
    plt.legend(loc='best')
    plt.title('t-SNE Visualization of Audio Embeddings')
    plt.tight_layout()
    plt.savefig('embedding_clusters.png')
    plt.show()
    
    return embeddings, final_labels, final_files, embeddings_2d

# Main test function for embeddings
if __name__ == "__main__":
    # Path to your model
    MODEL_PATH = "crnn_command_model.keras"
    
    # Path to your audio dataset (for testing)
    AUDIO_DIR = "./Audio_dataset"
    
    # Load model and metadata
    model, classes, max_duration, sample_rate, n_mels, n_fft, hop_length = load_model_and_metadata(MODEL_PATH)
    
    # Print model summary
    model.summary()
    
    # Create embedding model
    embedding_model = create_embedding_model(model)
    
    # Print embedding model summary
    print("\nEmbedding Model Summary:")
    embedding_model.summary()
    
    # Test specific audio file
    test_file = "./Audio_dataset/Aaj ka schedule batao/Aaj ka schedule batao_9b953e7b-86a8-42f0-b625-1434fb15392b_hi.wav"
    
    if os.path.exists(test_file):
        print(f"\nExtracting embedding from {test_file}")
        
        # Extract embedding
        embedding = extract_audio_embedding(
            model=embedding_model,
            audio_path=test_file,
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            max_duration=max_duration
        )
        
        print(f"Embedding shape: {embedding.shape}")
        
        # Find similar audios
        print("\nFinding similar audios...")
        similar_audios = find_similar_audios(
            reference_embedding=embedding,
            audio_directory=AUDIO_DIR,
            embedding_model=embedding_model,
            top_n=5,
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            max_duration=max_duration
        )
        
        # Print results
        print("\nSimilar audios:")
        for i, result in enumerate(similar_audios):
            print(f"{i+1}. {os.path.basename(result['file'])}")
            print(f"   Class: {result['class']}")
            print(f"   Similarity: {result['similarity']:.4f}")
    
    # Visualize embedding clusters
    print("\nVisualizing embedding clusters...")
    embeddings, labels, files, embeddings_2d = visualize_embedding_clusters(
        embedding_model=embedding_model,
        audio_directory=AUDIO_DIR,
        classes=classes[:10],  # Limit to first 10 classes for visualization
        n_samples_per_class=5,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        max_duration=max_duration
    )