import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import librosa
import json
import glob
import keras

# Import the CRNN class from your updated file
from Models import CRNN, load_and_preprocess_audio_file

def load_model_and_metadata(model_path, metadata_path=None):
    """
    Load the trained model and its metadata
    """
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

def predict_audio_class(model, audio_path, class_names, sample_rate=16000, n_mels=40, n_fft=1024, hop_length=512, max_duration=None):
    """
    Predict the class of an audio file
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
    
    # Get prediction
    prediction = model.predict(mel_spec, verbose=0)
    
    # Get top 3 predictions
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'class': class_names[idx],
            'confidence': float(prediction[0][idx])
        })
    
    return results, prediction[0]

def predict_audio_embeddings(model, audio_path, class_names, sample_rate=16000, n_mels=40, n_fft=1024, hop_length=512, max_duration=None):
    """
    Predict the class of an audio file
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
    
    # Get prediction
    prediction = model.predict(mel_spec, verbose=0)
    return prediction

def test_model_consistency(model, audio_directory, class_names, num_tests=5, **kwargs):
    """
    Test model consistency by predicting the same audio file multiple times
    """
    # Get all audio files in the directory
    audio_files = []
    for root, dirs, files in os.walk(audio_directory):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.flac'):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print("No audio files found in the directory")
        return
    
    # Select a few random files
    import random
    selected_files = random.sample(audio_files, min(num_tests, len(audio_files)))
    
    # Test each file multiple times
    for file_path in selected_files:
        print(f"\nTesting consistency for {os.path.basename(file_path)}")
        
        # Get the true class from the file path
        true_class = os.path.basename(os.path.dirname(file_path))
        print(f"True class: {true_class}")
        
        # Predict multiple times
        all_predictions = []
        for i in range(5):  # Test 5 times
            results, raw_probs = predict_audio_class(model, file_path, class_names, **kwargs)
            top_class = results[0]['class']
            confidence = results[0]['confidence']
            all_predictions.append((top_class, confidence))
            
            print(f"Run {i+1}: Predicted {top_class} with confidence {confidence:.4f}")
        
        # Check consistency
        predictions = [p[0] for p in all_predictions]
        if len(set(predictions)) == 1:
            print("✅ CONSISTENT - All predictions are the same")
        else:
            print("❌ INCONSISTENT - Predictions vary between runs")

# Main test function
if __name__ == "__main__":
    # Path to your model
    MODEL_PATH = "crnn_command_model.keras"
    
    # Path to your audio dataset (for testing)
    AUDIO_DIR = "./Audio_dataset"
    
    # Load model and metadata
    model, classes, max_duration, sample_rate, n_mels, n_fft, hop_length = load_model_and_metadata(MODEL_PATH)
    
    # Print model summary
    model.summary()
    
    # Print loaded classes
    print(f"Number of classes: {len(classes)}")
    print(f"First few classes: {classes[:5]}")
    
    # # Test specific audio file
    test_file = "./Audio_dataset/Aaj ka schedule batao/Aaj ka schedule batao_9b953e7b-86a8-42f0-b625-1434fb15392b_hi.wav"
    
    if os.path.exists(test_file):
        print(f"\nTesting with {test_file}")
        
        results, _ = predict_audio_class(
            model=model,
            audio_path=test_file,
            class_names=classes,
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            max_duration=max_duration
        )
        
        for i, result in enumerate(results):
            print(f"Top {i+1}: {result['class']} - {result['confidence']:.4f}")
        # print(results)
    # Test model consistency
    print("\nTesting model consistency across multiple runs...")
    test_model_consistency(
        model=model,
        audio_directory=AUDIO_DIR,
        class_names=classes,
        num_tests=3,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        max_duration=max_duration
    )