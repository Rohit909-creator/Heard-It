import os
import glob
import torch
import numpy as np
import torchaudio
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

def preprocess_audio_dataset(audio_dir, sample_rate=16000, n_mels=40, n_fft=1024, hop_length=512, max_duration=None, cache_dir=None):
    """
    Preprocess audio files in a directory into mel spectrograms using PyTorch and torchaudio.
    
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
    max_duration : float
        Maximum duration in seconds
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
                waveform, sr = torchaudio.load(file_path)
                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample if needed
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, sample_rate)
                    waveform = resampler(waveform)
                
                durations.append(waveform.shape[1] / sample_rate)
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
            # Load and preprocess audio using the same function we'll use for inference
            mel_spec = load_and_preprocess_audio_file(
                file_path, 
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                max_duration=max_duration,
                add_batch_dim=False
            )
            
            X.append(mel_spec)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert to numpy array
    X = np.array(X)
    
    # Save to cache if requested
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(torch.tensor(X), os.path.join(cache_dir, 'X.pt'))
        torch.save(torch.tensor(y), os.path.join(cache_dir, 'y.pt'))
        torch.save(torch.tensor(max_length), os.path.join(cache_dir, 'max_length.pt'))
        torch.save(torch.tensor(max_duration), os.path.join(cache_dir, 'max_duration.pt'))
        with open(os.path.join(cache_dir, 'classes.txt'), 'w') as f:
            for cls in label_encoder.classes_:
                f.write(f"{cls}\n")
    
    return X, y, label_encoder, max_length, max_duration

def load_and_preprocess_audio_file(audio_path, sample_rate=16000, n_mels=40, n_fft=1024, hop_length=512, max_duration=None, add_batch_dim=False):
    """
    Load and preprocess a single audio file for training or inference using PyTorch and torchaudio.
    Using the same function for both training and inference ensures consistency.
    """
    # Load audio file
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Squeeze extra dimension for processing
    waveform = waveform.squeeze(0)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Trim or pad if max_duration is specified
    if max_duration is not None:
        max_samples = int(max_duration * sample_rate)
        if waveform.shape[0] < max_samples:
            # Pad with zeros
            padding = max_samples - waveform.shape[0]
            waveform = F.pad(waveform, (0, padding))
        else:
            # Trim
            waveform = waveform[:max_samples]
    
    # Create mel spectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    
    # Extract mel spectrogram
    mel_spec = mel_spectrogram(waveform)
    
    # Convert to log scale (dB)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    # Convert to numpy for consistency with the original function
    mel_spec = mel_spec.numpy()
    
    # Normalize
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    
    # Transpose to have time as first dimension
    mel_spec = mel_spec.T
    
    # Add batch dimension if requested (for inference)
    if add_batch_dim:
        mel_spec = np.expand_dims(mel_spec, axis=0)
    
    return mel_spec

def load_and_preprocess_audio(waveform, sr, sample_rate=16000, n_mels=40, n_fft=1024, hop_length=512, max_duration=None, add_batch_dim=False):
    """
    Load and preprocess a single audio file for training or inference using PyTorch and torchaudio.
    Using the same function for both training and inference ensures consistency.
    """
    # Load audio file
    # waveform, sr = torchaudio.load(audio_path)
    waveform = torch.tensor(waveform)
    # Convert to mono if needed
    # print(waveform.shape, waveform.shape[0])
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # print(waveform.shape, waveform.shape[0])
    # Squeeze extra dimension for processing
    # waveform = waveform.squeeze(0)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Trim or pad if max_duration is specified
    if max_duration is not None:
        max_samples = int(max_duration * sample_rate)
        # print(waveform.shape, waveform.shape[0])
        if waveform.shape[0] < max_samples:
            # Pad with zeros
            padding = max_samples - waveform.shape[0]
            waveform = F.pad(waveform, (0, padding))
        else:
            # Trim
            waveform = waveform[:max_samples]
    
    # Create mel spectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    
    # Extract mel spectrogram
    mel_spec = mel_spectrogram(waveform)
    
    # Convert to log scale (dB)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    # Convert to numpy for consistency with the original function
    mel_spec = mel_spec.numpy()
    
    # Normalize
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    
    # Transpose to have time as first dimension
    mel_spec = mel_spec.T
    
    # Add batch dimension if requested (for inference)
    if add_batch_dim:
        mel_spec = np.expand_dims(mel_spec, axis=0)
    
    return mel_spec


# Example usage for PyTorch dataset creation
class AudioMelDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Function to load cached dataset
def load_cached_dataset(cache_dir):
    X = torch.load(os.path.join(cache_dir, 'X.pt')).numpy()
    y = torch.load(os.path.join(cache_dir, 'y.pt')).numpy()
    max_length = torch.load(os.path.join(cache_dir, 'max_length.pt')).item()
    max_duration = torch.load(os.path.join(cache_dir, 'max_duration.pt')).item()
    
    # Load class labels
    with open(os.path.join(cache_dir, 'classes.txt'), 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(classes)
    
    return X, y, label_encoder, max_length, max_duration

if __name__ == "__main__":
    
    # Preprocess the dataset
    X, y, label_encoder, max_length, max_duration = preprocess_audio_dataset(
        audio_dir="Audio_dataset", 
        cache_dir="./dataset_cache"
    )

    # Create dataset and dataloader
    dataset = AudioMelDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4
    )