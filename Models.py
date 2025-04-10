import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import librosa
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class CRNN(keras.Model):
    
    def __init__(self, mel_bins=40, time_frames=100, num_classes=10):
        super().__init__()
        
        # CNN layers for feature extraction
        self.conv1 = keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.ReLU()
        self.pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = keras.layers.Dropout(0.2)
        
        self.conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        self.act2 = keras.layers.ReLU()
        self.pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = keras.layers.Dropout(0.2)
        
        self.conv3 = keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same')
        self.bn3 = keras.layers.BatchNormalization()
        self.act3 = keras.layers.ReLU()
        self.pool3 = keras.layers.MaxPooling2D(pool_size=(1, 2))
        self.dropout3 = keras.layers.Dropout(0.2)
        
        # Reshape for RNN
        self.reshape = keras.layers.Reshape((-1, 64 * (mel_bins // 8)))
        
        # RNN layers for sequence modeling
        self.gru1 = keras.layers.GRU(64, return_sequences=True)
        self.dropout4 = keras.layers.Dropout(0.2)
        self.gru2 = keras.layers.GRU(64)
        self.dropout5 = keras.layers.Dropout(0.2)
        
        # Output layer
        self.dense = keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, x, training=False):
        # Expand dimensions for CNN (add channel dimension)
        x = tf.expand_dims(x, -1)  # Shape: [batch, time, mel, 1]
        
        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.dropout3(x, training=training)
        
        # Reshape for RNN layers
        x = self.reshape(x)
        
        # RNN sequence modeling
        x = self.gru1(x)
        x = self.dropout4(x, training=training)
        x = self.gru2(x)
        x = self.dropout5(x, training=training)
        
        # Output layer
        x = self.dense(x)
        
        return x

def preprocess_audio_dataset(audio_dir, sample_rate=16000, n_mels=40, n_fft=1024, hop_length=512, max_duration=None, cache_dir=None):
    """
    Preprocess audio files in a directory into mel spectrograms.
    
    Parameters:
    -----------
    audio_dir : str
        Directory containing audio files organized in subdirectories by class
    sample_rate : int
        Target sample rate for audio files
    n_mels : int
        Number of mel bands
    n_fft : int
        FFT window size
    hop_length : int
        Hop length for STFT
    max_duration : float or None
        Maximum duration in seconds. If None, will be determined automatically
    cache_dir : str or None
        If provided, will save processed features to this directory
        
    Returns:
    --------
    X : ndarray
        Mel spectrograms of shape (n_samples, time_frames, n_mels)
    y : ndarray
        Labels for each sample
    label_encoder : LabelEncoder
        Encoder for converting between numeric and string labels
    max_length : int
        Maximum time frames in the dataset
    """
    # Find all audio files
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg']
    audio_files = []
    labels = []
    
    # Get all class folders
    class_dirs = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
    
    # Collect all audio files and their labels
    for class_name in class_dirs:
        class_path = os.path.join(audio_dir, class_name)
        for ext in audio_extensions:
            files = glob.glob(os.path.join(class_path, ext))
            audio_files.extend(files)
            labels.extend([class_name] * len(files))
    
    print(f"Found {len(audio_files)} audio files in {len(class_dirs)} classes")
    
    # Determine max duration if not provided
    if max_duration is None:
        print("Finding maximum audio duration...")
        durations = []
        for i, file_path in enumerate(audio_files):
            if i % 100 == 0:
                print(f"Processing file {i+1}/{len(audio_files)} for duration calculation")
            try:
                audio, _ = librosa.load(file_path, sr=sample_rate, res_type='kaiser_fast')
                durations.append(len(audio) / sample_rate)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        max_duration = max(durations)
        print(f"Maximum audio duration: {max_duration:.2f} seconds")
    
    # Calculate maximum frame length
    max_length = int(np.ceil(max_duration * sample_rate / hop_length))
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # Process audio files
    X = []
    for i, file_path in enumerate(audio_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(audio_files)}")
        
        try:
            # Load audio file
            audio, _ = librosa.load(file_path, sr=sample_rate, res_type='kaiser_fast')
            
            # Pad or trim audio to max duration
            if len(audio) < int(max_duration * sample_rate):
                audio = np.pad(audio, (0, int(max_duration * sample_rate) - len(audio)))
            else:
                audio = audio[:int(max_duration * sample_rate)]
            
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
            
            X.append(mel_spec)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert to numpy array
    X = np.array(X)
    
    # Save to cache if requested
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, 'X.npy'), X)
        np.save(os.path.join(cache_dir, 'y.npy'), y)
        np.save(os.path.join(cache_dir, 'max_length.npy'), max_length)
        with open(os.path.join(cache_dir, 'classes.txt'), 'w') as f:
            for cls in label_encoder.classes_:
                f.write(f"{cls}\n")
    
    return X, y, label_encoder, max_length

def load_or_process_dataset(audio_dir, cache_dir, force_reprocess=False, **kwargs):
    """Load dataset from cache or process it if needed"""
    if not force_reprocess and cache_dir and os.path.exists(os.path.join(cache_dir, 'X.npy')):
        print("Loading dataset from cache...")
        X = np.load(os.path.join(cache_dir, 'X.npy'))
        y = np.load(os.path.join(cache_dir, 'y.npy'))
        max_length = np.load(os.path.join(cache_dir, 'max_length.npy')).item()
        
        # Load label encoder
        with open(os.path.join(cache_dir, 'classes.txt'), 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(classes)
        
        print(f"Loaded dataset with {X.shape[0]} samples")
        return X, y, label_encoder, max_length
    else:
        print("Processing audio dataset...")
        return preprocess_audio_dataset(audio_dir, cache_dir=cache_dir, **kwargs)

def train_model(audio_dir, model_save_path, cache_dir='./dataset_cache', epochs=50, batch_size=32, 
                sample_rate=16000, n_mels=40, validation_split=0.2, early_stopping_patience=10,
                max_duration=None, n_fft=1024, hop_length=512):
    """Train the CRNN model on the audio dataset"""
    
    # Process or load dataset
    X, y, label_encoder, max_length = load_or_process_dataset(
        audio_dir, 
        cache_dir, 
        sample_rate=sample_rate, 
        n_mels=n_mels,
        n_fft=n_fft, 
        hop_length=hop_length,
        max_duration=max_duration
    )
    
    # Get number of classes
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, stratify=y, random_state=42
    )
    
    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes=num_classes)
    
    # Initialize model
    model = CRNN(mel_bins=n_mels, time_frames=max_length, num_classes=num_classes)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False
        )
    ]
    
    # Build model by calling it once
    model.build(input_shape=(None, max_length, n_mels))
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return model, label_encoder, history

def convert_to_tflite(model, save_path, quantize=True):
    """Convert Keras model to TFLite for mobile deployment"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Apply quantization for smaller model size
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    tflite_model = converter.convert()
    
    # Save model
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    
    # Print model size
    model_size = os.path.getsize(save_path) / 1024 / 1024  # Convert to MB
    print(f"TFLite model size: {model_size:.2f} MB")
    
    return model_size

# Example usage
if __name__ == "__main__":
    # Replace with your audio dataset directory
    AUDIO_DIR = "./audio_dataset"
    
    # Train model
    model, label_encoder, history = train_model(
        audio_dir=AUDIO_DIR,
        model_save_path="./crnn_command_model.h5",
        cache_dir="./dataset_cache",
        epochs=50,
        batch_size=32,
        sample_rate=16000,
        n_mels=40,
        validation_split=0.2,
        early_stopping_patience=10
    )
    
    # Convert to TFLite for mobile deployment
    tflite_model_size = convert_to_tflite(
        model=model,
        save_path="./crnn_command_model.tflite",
        quantize=True
    )
    
    print(f"Training complete. Model saved as both H5 and TFLite formats.")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")