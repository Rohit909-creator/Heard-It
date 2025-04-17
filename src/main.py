import pyaudio
import numpy as np
import threading
import queue
import time
from typing import Optional, Dict
import json
import os
from Trainer import CRNN
from Inference import Matcher
from Preprocessing import load_and_preprocess_audio_file, load_and_preprocess_audio
import tensorflow as tf
from Utils import EnhancedSimilarityMatcher
import torch
checkpoint_path = "./lightning_logs/version_0/checkpoints/epoch=29-step=390.ckpt"

class SimpleMicStream:
    """Handles real-time audio capture from microphone"""
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_running = False
        
    def start_stream(self):
        """Start capturing audio from microphone"""
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.is_running = True
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def getFrame(self) -> np.ndarray:
        """Get latest audio frame from queue"""
        if not self.is_running:
            return None
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
        
    def stop_stream(self):
        """Stop and clean up audio stream"""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

class HotwordDetector:
    """Wake word detector using similarity-based detection"""
    def __init__(self, 
                 hotword: str,
                 reference_file: str,
                 model_path: str,
                 matcher,
                 threshold: float = 0.50,
                 window_length: float = 5.63):
        self.hotword = hotword
        self.threshold = threshold
        self.window_length = window_length
        self.sample_rate = 16000
        self.window_samples = int(self.window_length * self.sample_rate)
        
        # Load reference embeddings
        self.reference_embeddings = self._load_reference_embeddings(reference_file)
        
        # Initialize model
        self.model = CRNN.load_from_checkpoint(checkpoint_path, num_classes=74).to('cpu')
        
        self.matcher = matcher
        # Buffer for collecting audio frames
        self.audio_buffer = np.array([], dtype=np.float32)
        
    def _load_reference_embeddings(self, reference_file: str) -> np.ndarray:
        """Load reference embeddings from JSON file"""
        with open(reference_file, 'r') as f:
            data = json.load(f)
        return np.array(data['embeddings'])
    
    def scoreFrame(self, frame: np.ndarray) -> Optional[Dict]:
        """Process audio frame and check for wake word"""
        if frame is None:
            return None
            
        # Add frame to buffer
        self.audio_buffer = np.append(self.audio_buffer, frame)
        print("Time it")
        # If we have enough audio, process it
        if len(self.audio_buffer) >= self.window_samples:
            # Take the last window_samples
            audio_window = self.audio_buffer[-self.window_samples:]
            mel_spec = load_and_preprocess_audio(audio_window, sr=16000, max_duration=5.63)
            
            mel_spec = torch.tensor([mel_spec], dtype=torch.float32).unsqueeze(1)
            # Get embeddings for current audio
            current_embeddings = self.model(mel_spec)
            # current_embeddings = current_embeddings.squeeze().detach().numpy()
            # Use the matcher to determine if this is a wake word
            noise_level = self.matcher.estimate_noise_level(audio_window)
            
            is_wake_word, confidence, similarities = self.matcher.is_wake_word(current_embeddings, noise_level)
            
            # Set input tensor
            # interpreter.set_tensor(input_details[0]['index'], 
            #                     tf.reshape(current_embeddings, [1, -1]).numpy())
            # interpreter.set_tensor(input_details[1]['index'], 
            #                     np.array(noise_level, dtype=np.float32))
            
            # # # Run inference
            # interpreter.invoke()
            
            # # # Get output tensor
            # tflite_is_wake = interpreter.get_tensor(output_details[0]['index'])
            # tflite_score = interpreter.get_tensor(output_details[1]['index'])
            
            # # print(f"TFLite model output - Is wake word: {tflite_is_wake}, Score: {tflite_score}")
            
            
            
            # Trim buffer to prevent memory growth
            self.audio_buffer = self.audio_buffer[-self.window_samples:]
            
            # Check if confidence exceeds threshold
            if is_wake_word and confidence >= self.threshold:
                return {
                    "match": True,
                    "confidence": float(confidence),
                    # "similarities": similarities
                }
            
            # if tflite_is_wake and tflite_score >= self.threshold:
            #     return {
            #         "match": True,
            #         "confidence": float(tflite_score),
            #         "similarities": tflite_score
            #     }
                
        return {"match": False, "confidence": 0.0, "similarities": {}}

def main():
    
    import librosa
    from colorama import Fore, Style
    
    base_dir = "./"
    device = torch.device('cpu')
    model = CRNN.load_from_checkpoint(checkpoint_path, num_classes=74).to(device)
    
    
    # dir_list = os.listdir(os.path.join(base_dir, "wake_word_data", "recordings"))
    
    positive_files = [
        os.path.join(base_dir, r"Audio_dataset\Bolo\Bolo_9b953e7b-86a8-42f0-b625-1434fb15392b_hi.wav"),
        os.path.join(base_dir, r"Audio_dataset\Bolo\Bolo_28ca2041-5dda-42df-8123-f58ea9c3da00_hi.wav"),
        os.path.join(base_dir, r"Audio_dataset\Bolo\Bolo_bec003e2-3cb3-429c-8468-206a393c67ad_hi.wav"),
        os.path.join(base_dir, r"Audio_dataset\Bolo\Bolo_d088cdf6-0ef0-4656-aea8-eb9b004e82eb_hi.wav"),
        os.path.join(base_dir, r"Audio_dataset\Bolo\Bolo_e61a659d-56d5-4023-a499-1e1bccbc40e9_hi.wav"),
        os.path.join(base_dir, r"Audio_dataset\Bolo\Bolo_faf0731e-dfb9-4cfc-8119-259a79b27e12_hi.wav"),
        os.path.join(base_dir, r"Audio_dataset\Bolo\Bolo_fd2ada67-c2d9-4afe-b474-6386b87d8fc3_hi.wav"),
        # os.path.join(base_dir, r"tts_samples\positive\Nobita_en-US-zion.mp3"),
        # os.path.join(base_dir, "tts_samples", "negative", f"Aira1.mp3"),
        # os.path.join(base_dir, "tts_samples", "negative", f"Aira0.mp3"),
    ]
    
    negative_files = [
        # os.path.join(base_dir, r"Audio_dataset\Bolo\Bolo_bec003e2-3cb3-429c-8468-206a393c67ad_hi.wav"),
        # os.path.join(base_dir, r"Audio_dataset\Bolo\Bolo_d088cdf6-0ef0-4656-aea8-eb9b004e82eb_hi.wav"),
        os.path.join(base_dir, r"Audio_dataset\AC chalu karo\AC chalu karo_e61a659d-56d5-4023-a499-1e1bccbc40e9_hi.wav"),
        os.path.join(base_dir, r"Audio_dataset\AC chalu karo\AC chalu karo_faf0731e-dfb9-4cfc-8119-259a79b27e12_hi.wav"),
        os.path.join(base_dir, r"Audio_dataset\Anuplabdhata\Anuplabdhata_faf0731e-dfb9-4cfc-8119-259a79b27e12_hi.wav"),
        os.path.join(base_dir, r"Audio_dataset\Anuplabdhata\Anuplabdhata_28ca2041-5dda-42df-8123-f58ea9c3da00_hi.wav"),
        # os.path.join(base_dir, "tts_samples", "negative", "Xylophone_en-IN-aarav.mp3"),
        # os.path.join(base_dir, "tts_samples", "negative", "Xylophone_en-IN-alia.mp3"),
        # os.path.join(base_dir, "tts_samples", "negative", "Xylophone_en-US-zion.mp3"),
        # os.path.join(base_dir, "tts_samples", "negative", "Xylophone_en-US-natalie.mp3"),
        # os.path.join(base_dir, "tts_samples", "negative", "Quasar_en-IN-alia.mp3"),
        # os.path.join(base_dir, "tts_samples", "negative", "Quasar_en-IN-aarav.mp3"),
        # os.path.join(base_dir, "tts_samples", "negative", "Quasar_en-US-zion.mp3"),
        # os.path.join(base_dir, "tts_samples", "negative", "Quasar_en-US-natalie.mp3"),
    ]
        
    # Process positive examples
    print(f"{Fore.GREEN}Processing positive examples...{Style.RESET_ALL}")
    positive_embeddings = []
    # device = torch.device("cpu")
    for file in positive_files:
        
        mel_spec = load_and_preprocess_audio_file(file, max_duration=5.63)
        mel_spec = torch.tensor([mel_spec], dtype=torch.float32).unsqueeze(1)
        # audio, sr = librosa.load(file, sr=16000)
        # # Ensure audio is exactly 24000 samples long
        # expected_length = 24000
        # if len(audio) < expected_length:
        #     pad_length = expected_length - len(audio)
        #     audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros
        # audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(1)
        print(mel_spec.shape)
        emb = model(mel_spec)
        positive_embeddings.append(emb)
    
    # Process negative examples
    print(f"{Fore.RED}Processing negative examples...{Style.RESET_ALL}")
    negative_embeddings = []
    for file in negative_files:
        
        # audio, sr = librosa.load(file, sr=16000)
        # # Ensure audio is exactly 24000 samples long
        # expected_length = 24000
        # if len(audio) < expected_length:
        #     pad_length = expected_length - len(audio)
        #     audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros
        #     audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(1)
        mel_spec = load_and_preprocess_audio_file(file, max_duration=5.63)
        mel_spec = torch.tensor([mel_spec], dtype=torch.float32).unsqueeze(1)
        emb = model(mel_spec)
        negative_embeddings.append(emb)
    
    # Initialize matcher
    matcher = EnhancedSimilarityMatcher(positive_embeddings, negative_embeddings)
    print("ya here")
    
    # Initialize detector with your ONNX model path
    wake_word_detector = HotwordDetector(
        hotword="Bolo",
        reference_file="path_to_reference.json",  # Contains reference embeddings
        model_path="./resnet_50_arc/slim_93%_accuracy_72.7390%.onnx",
        matcher=matcher,
        window_length=5.63,
        threshold=0.5  # Adjust based on your needs
    )
    
    print("no yay here")
    # Start microphone stream
    mic_stream = SimpleMicStream()
    mic_stream.start_stream()
    
    print(f"Listening for wake word '{wake_word_detector.hotword}'...")
    try:
        while True:
            frame = mic_stream.getFrame()
            
            result = wake_word_detector.scoreFrame(frame)
            # print(frame)
            if result is None:
                continue
                
            if result["match"]:
                print(f"Wake word detected! (confidence: {result['confidence']:.2f})")
                # Add your callback action here
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic_stream.stop_stream()

if __name__ == "__main__":
    main()