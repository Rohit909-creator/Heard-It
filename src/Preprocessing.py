import os
import glob
import torch
import numpy as np
import torchaudio
import librosa
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from tqdm import tqdm

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
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.opus']
    audio_files = []
    labels = []
    
    # Get all class folders
    class_dirs = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
    # print(class_dirs)
    # Collect all audio files and their labels
    for class_name in class_dirs:
        class_path = os.path.join(audio_dir, class_name)
        for ext in audio_extensions:
            files = glob.glob(os.path.join(class_path, ext))
            audio_files.extend(files)
            # if class_name == "backward":
            #     print(files)
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
    waveform, sr = librosa.load(audio_path, sr=None)
    waveform = torch.from_numpy(waveform)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    # print("In load audio from file: ", waveform.shape, sr)
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # print(waveform.shape, sr)
    # Squeeze extra dimension for processing
    # print(waveform.shape, sr)
    waveform = waveform.squeeze(0)
    # print(f'After squeeze: {waveform.shape}')
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
    # print("In load audio: ",waveform.shape, sr)
    # print(waveform.shape, waveform.shape[0])
    # if waveform.shape[0] > 1:
    #     waveform = torch.mean(waveform, dim=0, keepdim=True)
    #     print(waveform.shape, waveform.shape[0])
    # Squeeze extra dimension for processing
    # print(f'After squeeze: {waveform.shape}')
    waveform = waveform.squeeze(0)
    
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


def audio_truncate(audio_dir):
    
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.opus']
    audio_files = []
    labels = []
    
    # Get all class folders
    class_dirs = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
    # print(class_dirs)
    # Collect all audio files and their labels
    for class_name in class_dirs:
        class_path = os.path.join(audio_dir, class_name)
        for ext in audio_extensions:
            files = glob.glob(os.path.join(class_path, ext))
            audio_files.extend(files)
            # if class_name == "backward":
            #     print(files)
            labels.extend([class_name] * len(files))
    
    print(f"Found {len(audio_files)} audio files in {len(class_dirs)} classes")
    
    for file_path in audio_files:
        
        audio, sr = torchaudio.load(file_path)
        if audio.shape[1] < 6000:
            print(f'Bad file:{file_path} length:{audio.shape[1]}')


def save_files2folders(audio_dir:str):
    import shutil
    
    # this is the first thing to run when you have to make dirs for this
    
    # audio_files = os.listdir(audio_dir)
    # audio_folders = [filename[:filename.index("_")] for filename in audio_files]
    # audio_folders_set = set(audio_folders)
    # print(audio_folders_set, len(audio_folders_set))
    
    # for audio_folder in audio_folders_set:
    #     os.makedirs(os.path.join(audio_dir, audio_folder), exist_ok=True)
    
    # after that ignore folders and focus on audio_files in the same directory
    
    audio_files = os.listdir(audio_dir)
    audio_files = [audio_file for audio_file in audio_files if ".wav" in audio_file]
    # print(audio_files)
    for audio_file in audio_files:
        src_path = os.path.join("AI_Audios_Augmented", audio_file)
        print(audio_file)
        dest_path = os.path.join("AI_Audios_Augmented", audio_file[:audio_file.index("_")])
        print(src_path, dest_path)
        shutil.move(src_path, dest_path)
        # break
    
def check4foldersize(audio_dir:str, audio_dir2:str):
    
    import shutil
    
    audio_folders = os.listdir(audio_dir)
    audio_folder_paths = []
    # audio_files = [audio_file for audio_file in audio_files if ".opus" in audio_file]
    # print(audio_folders)
    for audio_folder in audio_folders:
        
        audio_files = os.listdir(os.path.join(audio_dir, audio_folder))
        
        if len(audio_files) > 3000 and len(audio_files) < 5050:
            print(f"{audio_folder}: {len(audio_files)}") 
            audio_folder_paths.append(os.path.join(audio_dir, audio_folder))       
        # for audio_file in audio_files:
        #     src_path = os.path.join(audio_dir, audio_file)
    
    for audio_folder_path in audio_folder_paths:
        dest_path = os.path.join(audio_dir2, os.path.basename(audio_folder_path))
        if not os.path.exists(dest_path):  # Avoid overwriting if already exists
            shutil.copytree(audio_folder_path, dest_path)
            
def check4foldersize(audio_dir:str):
        
    audio_folders = os.listdir(audio_dir)
    audio_folder_paths = []
    # audio_files = [audio_file for audio_file in audio_files if ".opus" in audio_file]
    # print(audio_folders)
    for audio_folder in audio_folders:
        
        audio_files = os.listdir(os.path.join(audio_dir, audio_folder))
        
        if len(audio_files) > 1000:
            print(f"{audio_folder}: {len(audio_files)}") 
            # audio_folder_paths.append(os.path.join(audio_dir, audio_folder))
            
                    
if __name__ == "__main__":
    
    # Preprocess the dataset
    X, y, label_encoder, max_length, max_duration = preprocess_audio_dataset(
        audio_dir="./Audio_dataset3", 
        cache_dir="./mswc3_cache",
        max_duration=1.0
    )
    
    # print(X.shape, y.shape)

    # # Create dataset and dataloader
    # dataset = AudioMelDataset(X, y)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, 
    #     batch_size=32, 
    #     shuffle=True, 
    #     num_workers=4
    # )
    # import librosa
    # audio, sr = librosa.load("./Audios4testing/sample_3.wav", sr=16000)
    # # print(audio.shape, sr)
    # mel_spec2 = load_and_preprocess_audio_file("./Audios4testing/sample_3.wav")
    # mel_spec1 = load_and_preprocess_audio(audio, sr)
    
    # audio_dir = r"C:\Users\Rohit Francis\Desktop\Codes\Datasets\AI Generated Audios"
    # audio_truncate(audio_dir)
    
    # audio_dir = "./AI_Audios_Augmented"
    # save_files2folders(audio_dir)
    
    # audio_dir = r"C:\Users\Rohit Francis\Desktop\Codes\Datasets\en\clips"
    # audio_dir2 = "./Audio_dataset3"
    # check4foldersize(audio_dir, audio_dir2)
    
    # audio_dir = "./Audio_dataset2"
    # check4foldersize(audio_dir)
    
# considered: 2046
# crystal: 2003
# dog: 2024
# door: 2034
# hot: 2034
# lake: 2040
# london: 2026
# omens: 2010
# outside: 2047
# tried: 2005
# wait: 2038
# week: 2042


# anything: 4460
# away: 5004
# being: 4187
# best: 4224
# children: 4306
# county: 4371
# desert: 4993
# every: 4626
# everything: 4534
# few: 4237
# great: 4253
# heart: 4317
# help: 4437
# himself: 4178
# large: 4219
# last: 4895
# let: 4298
# located: 4770
# money: 4175
# music: 4657
# named: 5044
# next: 4576
# own: 4584
# please: 4445
# really: 4726
# second: 4214
# seven: 4239
# six: 4812
# small: 4696
# station: 4076
# thing: 4207
# under: 4461
# war: 4961
# water: 5023
# without: 4437
# woman: 4739
# young: 4273



# air: 3149
# album: 3235
# almost: 3145
# already: 3444
# among: 3066
# answered: 3114
# anything: 4460
# away: 5004
# become: 3393
# being: 4187
# best: 4224
# black: 3390
# children: 4306
# college: 3042
# company: 3228
# county: 4371
# days: 3699
# desert: 4993
# different: 3478
# doing: 3270
# done: 3665
# early: 3631
# eight: 3389
# end: 3979
# englishman: 3100
# ever: 3342
# every: 4626
# everyone: 3270
# everything: 4534
# family: 3707
# father: 3492
# few: 4237
# fire: 3314
# following: 3092
# game: 3569
# girl: 3313
# give: 3797
# great: 4253
# group: 3763
# heard: 3422
# heart: 4317
# help: 4437
# himself: 4178
# idea: 3057
# important: 3133
# include: 3234
# knew: 3611
# language: 3295
# large: 4219
# last: 4895
# let: 4298
# line: 3153
# live: 3504
# located: 4770
# looked: 3950
# looking: 3402
# main: 3044
# mean: 3437
# might: 3825
# mind: 3179
# money: 4175
# morning: 3109
# music: 4657
# named: 5044
# national: 3038
# near: 3321
# next: 4576
# nine: 3243
# north: 3700
# number: 3607
# once: 3710
# open: 3042
# own: 4584
# party: 3298
# play: 3850
# played: 3385
# please: 4445
# point: 3321
# public: 3466
# put: 3307
# read: 3597
# really: 4726
# red: 3693
# river: 3735
# road: 3311
# second: 4214
# seemed: 3555
# served: 3034
# set: 3291
# seven: 4239
# sheep: 3275
# show: 3574
# since: 3480
# six: 4812
# small: 4696
# sometimes: 3009
# son: 3194
# soon: 3144
# south: 3741
# state: 3539
# station: 4076
# street: 3053
# sun: 3522
# system: 3275
# thing: 4207
# times: 3161
# today: 3988
# top: 3086
# under: 4461
# university: 3862
# until: 3073
# wanted: 3518
# war: 4961
# water: 5023
# well: 3409
# white: 3458
# wind: 3141
# within: 3258
# without: 4437
# woman: 4739
# yes: 3316
# young: 4273