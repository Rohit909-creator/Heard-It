import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import random
import numpy as np
from typing import Optional, List, Tuple, Union
import math
import os
from tqdm import tqdm

class AudioAugmentor:
    """A class containing various audio augmentation methods using torchaudio, compatible with Windows."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the AudioAugmentor.
        
        Args:
            sample_rate: The sample rate of the audio files (Hz)
        """
        self.sample_rate = sample_rate
        
    def time_stretch(self, 
                    waveform: torch.Tensor, 
                    stretch_factor_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """
        Apply time stretching to an audio waveform.
        
        Args:
            waveform: Audio tensor [channels, time]
            stretch_factor_range: Range for random stretch factor (min, max)
                                  Values < 1 speed up, values > 1 slow down
        
        Returns:
            Time-stretched audio tensor
        """
        stretch_factor = random.uniform(*stretch_factor_range)
        
        if stretch_factor == 1.0:
            return waveform
            
        # Using torchaudio's phase vocoder for time stretching
        # Convert to spectrogram
        n_fft = 1024
        hop_length = 256
        
        spec_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None  # Return complex spectrogram
        )
        
        spec = spec_transform(waveform)
        
        # Apply time stretch on complex spectrogram
        stretched_spec = F.phase_vocoder(
            spec, 
            stretch_factor,
            hop_length
        )
        
        # Convert back to waveform
        griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            power=1.0,
            n_iter=32
        )
        
        stretched_waveform = griffin_lim(torch.abs(stretched_spec))
        
        # Make sure the output is the same type as input
        stretched_waveform = stretched_waveform.type_as(waveform)
        
        return stretched_waveform
    
    def pitch_shift(self, 
                   waveform: torch.Tensor, 
                   shift_range: Tuple[float, float] = (-3.0, 3.0)) -> torch.Tensor:
        """
        Apply pitch shifting to an audio waveform without using SoX.
        This is a Windows-compatible implementation using resampling approach.
        
        Args:
            waveform: Audio tensor [channels, time]
            shift_range: Range of semitones to shift the pitch by (min, max)
        
        Returns:
            Pitch-shifted audio tensor
        """
        # Choose random pitch shift in semitones
        n_steps = random.uniform(*shift_range)
        
        if n_steps == 0:
            return waveform
            
        # Calculate pitch shift factor: 2^(n_steps/12)
        pitch_factor = 2 ** (n_steps / 12)
        
        # Step 1: Resample to change pitch and tempo together
        orig_freq = self.sample_rate
        new_freq = int(orig_freq * pitch_factor)
        
        resampler_down = T.Resample(orig_freq=orig_freq, new_freq=new_freq)
        resampled = resampler_down(waveform)
        
        # Step 2: Apply time stretching to restore original tempo
        # This effectively changes only the pitch
        time_stretch_factor = 1.0 / pitch_factor
        
        n_fft = 1024
        hop_length = 256
        
        # Convert to spectrogram
        spec_transform = T.Spectrogram(
            n_fft=n_fft, 
            hop_length=hop_length,
            power=None
        )
        spec = spec_transform(resampled)
        
        # Time stretch to compensate tempo change
        stretched_spec = F.phase_vocoder(
            spec,
            time_stretch_factor,
            hop_length
        )
        
        # Convert back to waveform
        griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            power=1.0,
            n_iter=32
        )
        
        pitched_waveform = griffin_lim(torch.abs(stretched_spec))
        
        # Resample back to original sample rate
        resampler_up = T.Resample(orig_freq=new_freq, new_freq=orig_freq)
        pitched_waveform = resampler_up(pitched_waveform)
        
        # Ensure output length matches input
        if pitched_waveform.shape[1] > waveform.shape[1]:
            pitched_waveform = pitched_waveform[:, :waveform.shape[1]]
        elif pitched_waveform.shape[1] < waveform.shape[1]:
            padding = waveform.shape[1] - pitched_waveform.shape[1]
            pitched_waveform = torch.nn.functional.pad(pitched_waveform, (0, padding))
        
        return pitched_waveform
    
    def add_background_noise(self, 
                            waveform: torch.Tensor, 
                            noise_waveform: torch.Tensor, 
                            snr_range: Tuple[float, float] = (5.0, 20.0)) -> torch.Tensor:
        """
        Add background noise to an audio waveform at a specified SNR.
        
        Args:
            waveform: Audio tensor [channels, time]
            noise_waveform: Noise tensor [channels, time]
            snr_range: Signal-to-noise ratio range in dB (min, max)
        
        Returns:
            Noisy audio tensor
        """
        # Choose random SNR from range
        snr_db = random.uniform(*snr_range)
        
        # Calculate signal and noise power
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise_waveform ** 2)
        
        # If noise is too short, repeat it
        if noise_waveform.shape[1] < waveform.shape[1]:
            num_repeats = int(np.ceil(waveform.shape[1] / noise_waveform.shape[1]))
            noise_waveform = noise_waveform.repeat(1, num_repeats)
        
        # Trim noise to match signal length
        noise_waveform = noise_waveform[:, :waveform.shape[1]]
        
        # Calculate scaling factor for noise based on SNR
        alpha = torch.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
        
        # Scale noise and add to signal
        scaled_noise = noise_waveform * alpha
        noisy_waveform = waveform + scaled_noise
        
        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(noisy_waveform))
        if max_val > 1.0:
            noisy_waveform = noisy_waveform / max_val
        
        return noisy_waveform
    
    def add_room_simulation(self, 
                           waveform: torch.Tensor, 
                           rt60_range: Tuple[float, float] = (0.1, 1.0)) -> torch.Tensor:
        """
        Simulate room acoustics with reverb effect.
        
        Args:
            waveform: Audio tensor [channels, time]
            rt60_range: Reverberation time range in seconds (min, max)
        
        Returns:
            Reverberant audio tensor
        """
        # Choose random RT60 from range
        rt60 = random.uniform(*rt60_range)
        
        # Simple convolution-based reverb simulation using exponential decay
        n_samples = waveform.shape[1]
        decay_length = int(rt60 * self.sample_rate)
        
        # Create exponential decay filter
        decay = torch.exp(torch.linspace(0, -8, decay_length)).unsqueeze(0)
        
        # Apply reverb through convolution
        reverb_waveform = F.convolve(waveform, decay.to(waveform.device))
        
        # Normalize and trim to original length
        reverb_waveform = reverb_waveform[:, :n_samples]
        max_val = torch.max(torch.abs(reverb_waveform))
        if max_val > 1.0:
            reverb_waveform = reverb_waveform / max_val
        
        # Mix original and reverberant signals
        mix_ratio = random.uniform(0.5, 0.9)
        result = mix_ratio * waveform + (1 - mix_ratio) * reverb_waveform
        
        return result
    
    def adjust_volume(self, 
                     waveform: torch.Tensor, 
                     gain_range: Tuple[float, float] = (-10.0, 3.0)) -> torch.Tensor:
        """
        Adjust the volume of an audio waveform.
        
        Args:
            waveform: Audio tensor [channels, time]
            gain_range: Gain adjustment range in dB (min, max)
        
        Returns:
            Volume-adjusted audio tensor
        """
        gain_db = random.uniform(*gain_range)
        gain_factor = 10 ** (gain_db / 20.0)  # Convert dB to amplitude factor
        
        return waveform * gain_factor
    
    def time_shift(self, 
                  waveform: torch.Tensor, 
                  shift_range: Tuple[float, float] = (-0.2, 0.2)) -> torch.Tensor:
        """
        Apply random time shifting to an audio waveform.
        
        Args:
            waveform: Audio tensor [channels, time]
            shift_range: Shift range as fraction of total length (min, max)
        
        Returns:
            Time-shifted audio tensor
        """
        n_samples = waveform.shape[1]
        shift_factor = random.uniform(*shift_range)
        shift_samples = int(shift_factor * n_samples)
        
        shifted_waveform = torch.roll(waveform, shifts=shift_samples, dims=1)
        
        # Zero out the rolled part for clean shifting
        if shift_samples > 0:
            shifted_waveform[:, :shift_samples] = 0
        elif shift_samples < 0:
            shifted_waveform[:, shift_samples:] = 0
            
        return shifted_waveform
    
    def spec_augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment-style augmentation with more robust implementation.
        
        Args:
            waveform: Audio tensor [channels, time]
            
        Returns:
            Augmented audio tensor
        """
        # We'll use a simpler approach - apply freq and time masking directly to the spectrogram
        # then convert back, without the problematic InverseMelScale transform
        
        # First convert to spectrogram
        n_fft = 512
        hop_length = 128
        
        # Create spectrogram
        spec_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0  # Power spectrogram
        )
        
        # Apply spectrogram transform
        spec = spec_transform(waveform)
        
        # Apply time and frequency masking directly to linear spectrogram
        time_masking = T.TimeMasking(time_mask_param=20)
        freq_masking = T.FrequencyMasking(freq_mask_param=15)
        
        # Apply masking
        masked_spec = time_masking(spec)
        masked_spec = freq_masking(masked_spec)
        
        # Convert back to waveform
        griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
            n_iter=32
        )
        
        augmented_waveform = griffin_lim(masked_spec)
        
        # Make sure the output is the same type and shape as input
        if augmented_waveform.shape[1] != waveform.shape[1]:
            # Pad or trim to match original length
            if augmented_waveform.shape[1] < waveform.shape[1]:
                pad_size = waveform.shape[1] - augmented_waveform.shape[1]
                augmented_waveform = torch.nn.functional.pad(augmented_waveform, (0, pad_size))
            else:
                augmented_waveform = augmented_waveform[:, :waveform.shape[1]]
        
        # Ensure we have the same number of channels
        if augmented_waveform.shape[0] != waveform.shape[0]:
            if augmented_waveform.shape[0] == 1 and waveform.shape[0] > 1:
                augmented_waveform = augmented_waveform.repeat(waveform.shape[0], 1)
        
        return augmented_waveform
        
    def combine_augmentations(self, 
                            waveform: torch.Tensor,
                            noise_files: Optional[List[str]] = None,
                            num_augmentations: int = 2) -> torch.Tensor:
        """Apply multiple random augmentations to an audio sample."""
        # available_augmentations = [
            # self.time_stretch,
            # self.pitch_shift,
            # self.time_shift,
            # self.adjust_volume,
            # self.add_room_simulation,
            # self.spec_augment
        # ]
        available_augmentations = []
        for i in range(len(noise_files)):
            noise_aug = lambda x: self._get_noise_augmentation(x, noise_files)
            available_augmentations.append(noise_aug)
        
        # Add noise augmentation if noise files provided
        # if noise_files and len(noise_files) > 0:
        #     noise_aug = lambda x: self._get_noise_augmentation(x, noise_files)
        #     available_augmentations.append(noise_aug)
        
        # Select random augmentations
        selected_augmentations = random.sample(
            available_augmentations, 
            min(num_augmentations, len(available_augmentations))
        )
        
        # Apply selected augmentations sequentially with error handling
        augmented_waveform = waveform
        for augment_fn in selected_augmentations:
            try:
                augmented_waveform = augment_fn(augmented_waveform)
            except Exception as e:
                print(f"Warning: Augmentation function {augment_fn.__name__} failed: {str(e)}")
                # Continue with the original waveform if an augmentation fails
                continue
                
        return augmented_waveform
    
    def _get_noise_augmentation(self, waveform: torch.Tensor, noise_files: List[str]) -> torch.Tensor:
        """Helper method to apply noise augmentation using a random noise file."""
        # Select random noise file
        noise_file = random.choice(noise_files)
        
        # Load noise waveform
        noise_waveform, _ = torchaudio.load(noise_file)
        
        # Match channels with input
        if noise_waveform.shape[0] != waveform.shape[0]:
            if noise_waveform.shape[0] == 1 and waveform.shape[0] == 2:
                noise_waveform = noise_waveform.repeat(2, 1)
            elif noise_waveform.shape[0] == 2 and waveform.shape[0] == 1:
                noise_waveform = torch.mean(noise_waveform, dim=0, keepdim=True)
        
        # Apply noise
        return self.add_background_noise(waveform, noise_waveform)


# Example usage functions
def load_audio_sample(file_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load an audio file and resample if necessary.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Audio tensor [channels, time]
    """
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Resample if needed
    if sample_rate != target_sr:
        resampler = T.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        
    return waveform

def augment_dataset(audio_files: List[str], 
                   noise_files: Optional[List[str]] = None,
                   output_dir: str = 'augmented/',
                   augmentations_per_file: int = 9,
                   sample_rate: int = 16000) -> None:
    """
    Augment an entire dataset of audio files.
    
    Args:
        audio_files: List of paths to audio files
        noise_files: List of paths to noise files (optional)
        output_dir: Directory to save augmented files
        augmentations_per_file: Number of augmented versions to create per file
        sample_rate: Target sample rate
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    augmentor = AudioAugmentor(sample_rate=sample_rate)
    
    for audio_file in tqdm(audio_files, "Augmenting"):
        base_name = os.path.basename(audio_file)
        name, ext = os.path.splitext(base_name)
        
        # Load audio
        waveform = load_audio_sample(audio_file, target_sr=sample_rate)
        
        # Create augmented versions
        for i in range(augmentations_per_file):
            augmented = augmentor.combine_augmentations(
                waveform, 
                noise_files=noise_files,
                num_augmentations=random.randint(1, 4)  # Random number of augmentations
            )
            
            # Save augmented audio
            output_path = os.path.join(output_dir, f"{name}_aug{i+1}{ext}")
            torchaudio.save(output_path, augmented, sample_rate)



# Here's an example of how to use the toolkit:
if __name__ == "__main__":
    import glob
    
    # Example usage
    audio_file = "noise_1.wav"
    noise_files = [
        "Noise1.wav",
        "Noise2.wav",
        "Noise3.wav",
        "noise_1.wav",
        "noise_2.wav",
        "noise_3.wav",
        "noise_4.wav",
        "noise_5.wav",
    ]
    
    # Load audio
    waveform = load_audio_sample(audio_file)
    
    # Initialize augmentor
    augmentor = AudioAugmentor()
    
    # Apply a specific augmentation
    time_stretched = augmentor.time_stretch(waveform)
    
    # Apply multiple random augmentations
    augmented = augmentor.combine_augmentations(waveform, noise_files=noise_files)
    
    # Process a dataset
    
    audio_dir = r"C:\Users\Rohit Francis\Desktop\Codes\Datasets\AI Generated Audios"
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
    # "C:\Users\Rohit Francis\Desktop\Codes\Datasets\AI Generated Audios\Sharanya\Sharanya_fd2ada67-c2d9-4afe-b474-6386b87d8fc3_hi.wav"
    print(f"Found {len(audio_files)} audio files in {len(class_dirs)} classes")
    audio_files = audio_files[:audio_files.index(r"C:\Users\Rohit Francis\Desktop\Codes\Datasets\AI Generated Audios\Sharanya\Sharanya_fd2ada67-c2d9-4afe-b474-6386b87d8fc3_hi.wav")]
    print(audio_files[-1])
    # audio_files = ["output.wav", "output2.wav", "output3.wav"]
    augment_dataset(audio_files, noise_files, output_dir="AI_Audios_Augmented/")

# # Here's an example of how to use the toolkit:
# if __name__ == "__main__":
#     # Example usage
#     audio_file = "Chutiya_28ca2041-5dda-42df-8123-f58ea9c3da00_hi.wav"
#     noise_files = [
#         "Noise1.wav",
#         "Noise2.wav",
#         "Noise3.wav"
#     ]
    
#     # Load audio
#     waveform = load_audio_sample(audio_file)
    
#     # Initialize augmentor
#     augmentor = AudioAugmentor()
    
#     # Apply a specific augmentation
#     time_stretched = augmentor.time_stretch(waveform)
    
#     # Apply multiple random augmentations
#     augmented = augmentor.combine_augmentations(waveform, noise_files=noise_files)
    
#     # Process a dataset
#     audio_files = ["Kya chal raha hai_28ca2041-5dda-42df-8123-f58ea9c3da00_hi.wav", "test2.wav"]
#     augment_dataset(audio_files, noise_files, output_dir="Augmented/")