import sounddevice as sd
import numpy as np
from scipy.io import wavfile

# Function to record audio samples
def record_samples(filename = 'sample', num_samples=3, duration=3, sr=16000):
    """
    Record audio samples from the user and save as WAV files.
    Args:
        num_samples (int): Number of samples to record (default: 3).
        duration (float): Duration of each recording in seconds (default: 3).
        sr (int): Sample rate (default: 16000 Hz).
    Returns:
        list: List of recorded audio samples as NumPy arrays.
    """
    print(f"Recording {num_samples} samples, each {duration} seconds long...")
    samples = []
    for i in range(num_samples):
        print(f"Sample {i+1}: Say your sentence now (recording starts in 1 second)...")
        sd.wait(4)  # 1-second delay to give the user time to prepare
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()  # Wait for recording to finish
        samples.append(audio.flatten())
        # Save the audio as a WAV file
        # Convert float32 audio to int16 for WAV format (standard for WAV files)
        audio_int16 = (audio.flatten() * 32767).astype(np.int16)
        wavfile.write(f"./Audios4testing/{filename}_{i+1}.wav", sr, audio_int16)
        print(f"Recording finished for sample {i+1}. Saved as 'sample_{i+1}.wav'.")
    return samples

# Main execution for this step
if __name__ == "__main__":
    # Prompt user for the sentence and number of samples
    target_sentence = input("Enter the word you want to use (e.g., 'Alexa', 'Aira'): ").strip().lower()
    num_samples = int(input("How many samples do you want to record (3-5 recommended)? "))
    if num_samples < 3 or num_samples > 5:
        print("Please choose between 3 and 5 samples.")
        num_samples = 3  # Default to 3 if input is invalid

    # Record the samples
    audio_samples = record_samples(filename=target_sentence, num_samples=num_samples,duration=1)

    # Save the samples and target sentence for later use
    np.save("audio_samples.npy", np.array(audio_samples, dtype=object), allow_pickle=True)
    with open("target_sentence.txt", "w") as f:
        f.write(target_sentence)

    print(f"Recorded {num_samples} samples for the sentence: '{target_sentence}'")
    print("Samples saved to 'audio_samples.npy' and sentence saved to 'target_sentence.txt'.")
    print("Individual WAV files saved as 'sample_1.wav', 'sample_2.wav', etc.")

    # Play back the first sample to verify (optional)
    print("Playing back the first sample...")
    sd.play(audio_samples[0], samplerate=16000)
    sd.wait()
    print("Playback finished.")