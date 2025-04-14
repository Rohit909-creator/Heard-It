import numpy as np
import librosa
import pyroomacoustics as pra
import soundfile as sf
import os
from scipy import signal
import random


class AudioAugmenter:
    """
    A class for applying various audio augmentation techniques to audio samples
    for improving speech recognition model performance.
    """
    
    def __init__(self, sample_rate=16000, seed=None):
        """
        Initialize the AudioAugmenter.
        
        Args:
            sample_rate (int): Sample rate of the audio files
            seed (int, optional): Random seed for reproducibility
        """
        self.sample_rate = sample_rate
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
    def load_audio(self, file_path):
        """Load audio file using librosa."""
        audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio
    
    def save_audio(self, audio, file_path):
        """Save audio to file."""
        sf.write(file_path, audio, self.sample_rate)
    
    def time_stretch(self, audio, rate_range=(0.8, 1.2)):
        """
        Stretch or compress the audio in time without changing pitch.
        
        Args:
            audio (np.ndarray): Input audio signal
            rate_range (tuple): Range of stretching factors (below 1: stretch, above 1: compress)
            
        Returns:
            np.ndarray: Time-stretched audio
        """
        rate = np.random.uniform(*rate_range)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio, semitone_range=(-4, 4)):
        """
        Shift the pitch of the audio.
        
        Args:
            audio (np.ndarray): Input audio signal
            semitone_range (tuple): Range of semitones to shift pitch by
            
        Returns:
            np.ndarray: Pitch-shifted audio
        """
        n_steps = np.random.uniform(*semitone_range)
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def add_background_noise(self, audio, noise_file, snr_range=(5, 20)):
        """
        Add background noise to the audio at a specified signal-to-noise ratio.
        
        Args:
            audio (np.ndarray): Input audio signal
            noise_file (str): Path to the noise audio file
            snr_range (tuple): Range of signal-to-noise ratios in dB
            
        Returns:
            np.ndarray: Noisy audio
        """
        # Load noise and ensure it's the same length as the audio
        noise = self.load_audio(noise_file)
        
        # Loop the noise if it's shorter than the audio
        if len(noise) < len(audio):
            noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
        
        # Trim or pad noise to match audio length
        noise = noise[:len(audio)] if len(noise) >= len(audio) else np.pad(noise, (0, len(audio) - len(noise)))
        
        # Calculate audio and noise power
        audio_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Set the SNR
        snr = np.random.uniform(*snr_range)
        noise_scale = np.sqrt(audio_power / (noise_power * 10 ** (snr / 10)))
        
        # Mix audio and noise
        noisy_audio = audio + noise_scale * noise
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val
            
        return noisy_audio
    
    def mix_random_noise_from_directory(self, audio, noise_dir, snr_range=(5, 20), prob=0.7):
        """
        Add a randomly selected background noise from a directory.
        
        Args:
            audio (np.ndarray): Input audio signal
            noise_dir (str): Directory containing noise audio files
            snr_range (tuple): Range of signal-to-noise ratios in dB
            prob (float): Probability of adding noise
            
        Returns:
            np.ndarray: Audio with or without noise
        """
        if np.random.random() > prob:
            return audio
            
        noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) 
                      if f.endswith(('.wav', '.mp3', '.flac'))]
        
        if not noise_files:
            return audio
            
        selected_noise = random.choice(noise_files)
        return self.add_background_noise(audio, selected_noise, snr_range)
    
    def apply_room_simulation(self, audio, room_dim_range=(3, 10), rt60_range=(0.2, 0.8)):
        """
        Apply room simulation effects (reverb/echo) to audio.
        
        Args:
            audio (np.ndarray): Input audio signal
            room_dim_range (tuple): Range for room dimensions in meters
            rt60_range (tuple): Range for reverberation time in seconds
            
        Returns:
            np.ndarray: Audio with room effects
        """
        # Create a random room
        x_dim = np.random.uniform(*room_dim_range)
        y_dim = np.random.uniform(*room_dim_range)
        z_dim = np.random.uniform(2.5, 4)
        
        # Set reverberation time
        rt60 = np.random.uniform(*rt60_range)
        
        # Create room with specified dimensions and RT60
        room = pra.ShoeBox(
            [x_dim, y_dim, z_dim],
            fs=self.sample_rate,
            materials=pra.Material(energy_absorption=0.2),
            max_order=17,
        )
        
        # Adjust the materials to achieve the desired RT60
        room.set_rt60(rt60)
        
        # Add a source somewhere in the room
        source_pos = [np.random.uniform(0.5, x_dim-0.5), 
                      np.random.uniform(0.5, y_dim-0.5), 
                      np.random.uniform(1.0, 1.8)]
        room.add_source(source_pos, signal=audio)
        
        # Add a microphone
        mic_pos = [np.random.uniform(0.5, x_dim-0.5), 
                   np.random.uniform(0.5, y_dim-0.5), 
                   np.random.uniform(1.0, 1.8)]
        room.add_microphone(mic_pos)
        
        # Compute the room impulse response
        room.compute_rir()
        
        # Simulate room
        room.simulate()
        
        # Get the audio from the microphone
        reverb_audio = room.mic_array.signals[0, :]
        
        # Normalize output
        reverb_audio = reverb_audio / np.max(np.abs(reverb_audio)) * np.max(np.abs(audio))
        
        # Trim to original length if needed
        if len(reverb_audio) > len(audio):
            reverb_audio = reverb_audio[:len(audio)]
        else:
            reverb_audio = np.pad(reverb_audio, (0, len(audio) - len(reverb_audio)))
            
        return reverb_audio
    
    def change_volume(self, audio, gain_range=(0.5, 1.5)):
        """
        Apply random volume change to audio.
        
        Args:
            audio (np.ndarray): Input audio signal
            gain_range (tuple): Range of gain factors to apply
            
        Returns:
            np.ndarray: Volume-adjusted audio
        """
        gain = np.random.uniform(*gain_range)
        audio_modified = audio * gain
        
        # Clip to prevent distortion
        audio_modified = np.clip(audio_modified, -1.0, 1.0)
        
        return audio_modified
    
    def add_random_frequency_filtering(self, audio, filter_types=None):
        """
        Apply random frequency filtering to simulate different device characteristics.
        
        Args:
            audio (np.ndarray): Input audio signal
            filter_types (list, optional): List of filter types to choose from
            
        Returns:
            np.ndarray: Filtered audio
        """
        if filter_types is None:
            filter_types = ['lowpass', 'highpass', 'bandpass']
            
        filter_type = random.choice(filter_types)
        nyquist = self.sample_rate // 2
        
        if filter_type == 'lowpass':
            # Random cutoff between 1kHz and Nyquist
            cutoff = np.random.uniform(1000, nyquist)
            b, a = signal.butter(4, cutoff / nyquist, btype='low')
            
        elif filter_type == 'highpass':
            # Random cutoff between 80Hz and 1kHz
            cutoff = np.random.uniform(80, 1000)
            b, a = signal.butter(4, cutoff / nyquist, btype='high')
            
        elif filter_type == 'bandpass':
            # Random bandwidth between 500Hz and 3kHz
            low_cutoff = np.random.uniform(200, 2000)
            high_cutoff = low_cutoff + np.random.uniform(500, 3000)
            high_cutoff = min(high_cutoff, nyquist - 100)  # Ensure it's below Nyquist
            b, a = signal.butter(2, [low_cutoff / nyquist, high_cutoff / nyquist], btype='band')
            
        # Apply the filter
        filtered_audio = signal.lfilter(b, a, audio)
        
        return filtered_audio
    
    def time_shift(self, audio, shift_range=(-0.1, 0.1)):
        """
        Shift audio in time by a random amount.
        
        Args:
            audio (np.ndarray): Input audio signal
            shift_range (tuple): Range of shift amount as a fraction of total length
            
        Returns:
            np.ndarray: Time-shifted audio
        """
        shift_factor = np.random.uniform(*shift_range)
        shift_samples = int(len(audio) * shift_factor)
        
        if shift_samples > 0:
            # Shift right (introduce silence at beginning)
            shifted_audio = np.concatenate([np.zeros(shift_samples), audio[:-shift_samples]])
        else:
            # Shift left (introduce silence at end)
            shift_samples = abs(shift_samples)
            shifted_audio = np.concatenate([audio[shift_samples:], np.zeros(shift_samples)])
            
        return shifted_audio
    
    def apply_random_augmentations(self, audio, noise_dir=None, prob_dict=None):
        """
        Apply multiple random augmentations with specified probabilities.
        
        Args:
            audio (np.ndarray): Input audio signal
            noise_dir (str, optional): Directory containing noise files
            prob_dict (dict, optional): Dictionary of augmentation probabilities
            
        Returns:
            np.ndarray: Augmented audio
        """
        # Default probabilities if not provided
        if prob_dict is None:
            prob_dict = {
                'time_stretch': 0.5,
                'pitch_shift': 0.5,
                'add_noise': 0.5 if noise_dir else 0,
                'room_simulation': 0.3,
                'volume_change': 0.7,
                'frequency_filter': 0.3,
                'time_shift': 0.5
            }
        
        augmented_audio = audio.copy()
        
        # Apply each augmentation based on probability
        if np.random.random() < prob_dict.get('time_stretch', 0):
            augmented_audio = self.time_stretch(augmented_audio)
            
        if np.random.random() < prob_dict.get('pitch_shift', 0):
            augmented_audio = self.pitch_shift(augmented_audio)
            
        if noise_dir and np.random.random() < prob_dict.get('add_noise', 0):
            augmented_audio = self.mix_random_noise_from_directory(
                augmented_audio, noise_dir, prob=1.0)  # prob=1.0 because we already checked
            
        if np.random.random() < prob_dict.get('room_simulation', 0):
            augmented_audio = self.apply_room_simulation(augmented_audio)
            
        if np.random.random() < prob_dict.get('volume_change', 0):
            augmented_audio = self.change_volume(augmented_audio)
            
        if np.random.random() < prob_dict.get('frequency_filter', 0):
            augmented_audio = self.add_random_frequency_filtering(augmented_audio)
            
        if np.random.random() < prob_dict.get('time_shift', 0):
            augmented_audio = self.time_shift(augmented_audio)
            
        return augmented_audio
    
    def generate_augmented_dataset(self, input_dir, output_dir, noise_dir=None, 
                                  augmentations_per_file=5, prob_dict=None):
        """
        Generate an augmented dataset from original audio files.
        
        Args:
            input_dir (str): Directory containing original audio files
            output_dir (str): Directory to save augmented files
            noise_dir (str, optional): Directory containing noise files
            augmentations_per_file (int): Number of augmented versions per original file
            prob_dict (dict, optional): Dictionary of augmentation probabilities
            
        Returns:
            list: Paths to all generated files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        audio_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                      if f.endswith(('.wav', '.mp3', '.flac'))]
        
        generated_files = []
        
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            base_name, ext = os.path.splitext(filename)
            
            # Load original audio
            audio = self.load_audio(audio_file)
            
            # Save original copy if needed
            # original_output = os.path.join(output_dir, filename)
            # self.save_audio(audio, original_output)
            # generated_files.append(original_output)
            
            # Generate augmented versions
            for i in range(augmentations_per_file):
                augmented_audio = self.apply_random_augmentations(audio, noise_dir, prob_dict)
                
                # Save augmented audio
                aug_filename = f"{base_name}_aug_{i+1}{ext}"
                aug_path = os.path.join(output_dir, aug_filename)
                self.save_audio(augmented_audio, aug_path)
                generated_files.append(aug_path)
                
        return generated_files


# Example usage:
if __name__ == "__main__":
    augmenter = AudioAugmenter(sample_rate=16000)
    
    # Example 1: Single file augmentation
    audio = augmenter.load_audio("test.wav")
    
    # Individual augmentations
    stretched_audio = augmenter.time_stretch(audio)
    pitched_audio = augmenter.pitch_shift(audio)
    
    # Apply room simulation
    reverb_audio = augmenter.apply_room_simulation(audio)
    
    # Add background noise (if you have a noise file)
    # noisy_audio = augmenter.add_background_noise(audio, "noise.wav")
    
    # Apply all augmentations randomly with default probabilities
    augmented_audio = augmenter.apply_random_augmentations(
        audio, noise_dir="path/to/noise_files")
    
    # Example 2: Generate an augmented dataset
    # generated_files = augmenter.generate_augmented_dataset(
    #     "input_audio", "augmented_output", "noise_samples", 
    #     augmentations_per_file=3)