import pyaudio
import numpy as np
import threading
import queue
import time
from typing import Optional, Dict
import json
import os
from Trainer import ResNetMel, ResNetMelLite
# from Inference import Matcher
from Preprocessing import load_and_preprocess_audio_file, load_and_preprocess_audio
import tensorflow as tf
from Utils import Matcher, EnhancedSimilarityMatcher, fetch_audios
import torch
from colorama import Fore, Style
# checkpoint_path = "./lightning_logs/version_23/checkpoints/epoch=14-step=46560.ckpt"
# checkpoint_path = "./lightning_logs/version_27/checkpoints/epoch=9-step=31040.ckpt"
# checkpoint_path = "./lightning_logs/version_27/checkpoints/epoch=14-step=46560.ckpt"
# checkpoint_path = "./lightning_logs/version_25/checkpoints/epoch=13-step=44142.ckpt"
# checkpoint_path = "./lightning_logs/version_26/checkpoints/epoch=24-step=78825.ckpt"

# 269 classes model
# checkpoint_path = "./lightning_logs/version_29/checkpoints/epoch=9-step=34120.ckpt"
# checkpoint_path = "./lightning_logs/version_29/checkpoints/epoch=6-step=23884.ckpt"
# checkpoint_path = "./lightning_logs/version_29/checkpoints/epoch=9-step=34120.ckpt"
# checkpoint_path = "./lightning_logs/version_30/checkpoints/epoch=19-step=68240.ckpt"
# classes_path = "./mswc3_cache/classes.txt"

# 123 classes version stable one
checkpoint_path = "./lightning_logs/version_31/checkpoints/epoch=11-step=136404.ckpt"

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
                 threshold: float = 0.5,
                 window_length: float = 1.0):
        self.hotword = hotword
        self.threshold = threshold
        self.window_length = window_length
        self.sample_rate = 16000
        self.window_samples = int(self.window_length * self.sample_rate)
        
        # Load reference embeddings
        self.reference_embeddings = self._load_reference_embeddings(reference_file)
        
        # Initialize model
        # self.model = CRNN.load_from_checkpoint(checkpoint_path, num_classes=74).to('cpu')
        # self.model = ResNetMel.load_from_checkpoint(checkpoint_path, num_classes=52).to('cpu')
        self.model = ResNetMelLite.load_from_checkpoint(checkpoint_path, num_classes=123).to('cpu')
        # self.model = ResNetMel.load_from_checkpoint(checkpoint_path, num_classes=387).to('cpu')
        
        
        # 269 class version 
        # self.model = ResNetMelLite.load_from_checkpoint(checkpoint_path, num_classes=269).to('cpu')
        
        self.model.model.fc[4] = torch.nn.Sequential()
        self.model.eval()
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
        # print("Time it")
        # If we have enough audio, process it
        if len(self.audio_buffer) >= self.window_samples:
            # Take the last window_samples
            audio_window = self.audio_buffer[-self.window_samples:]
            # print(audio_window.shape)
            mel_spec = load_and_preprocess_audio(audio_window, sr=16000, max_duration=1.0)
            
            mel_spec_tensor = torch.tensor([mel_spec], dtype=torch.float32)
            mel_spec_tensor = mel_spec_tensor.unsqueeze(1)
            mel_spec_tensor = mel_spec_tensor
            # print(mel_spec_tensor.shape)
            current_embeddings = self.model(mel_spec_tensor)
            # Get embeddings for current audio
            # current_embeddings = self.model(mel_spec)
            # current_embeddings = current_embeddings.squeeze().detach().numpy()
            # Use the matcher to determine if this is a wake word
            # print(torch.cosine_similarity(current_embeddings, out1, dim=-1))
            reference_embeddings = torch.tensor(self.reference_embeddings, dtype=torch.float32)
            # is_wake_word, confidence = self.matcher.match(current_embeddings, reference_embeddings)
            # print(is_wake_word, confidence)
            noise_level = self.matcher.estimate_noise_level(audio_window)
            
            is_wake_word, confidence, similarities = self.matcher.is_wake_word(current_embeddings, noise_level)
            
            
            
            # Trim buffer to prevent memory growth
            self.audio_buffer = self.audio_buffer[-self.window_samples:]
            
            # Check if confidence exceeds threshold
            if is_wake_word and confidence >= self.threshold:
                return {
                    "match": True,
                    "confidence": float(confidence),
                    # "similarities": similarities
                }
                
        return {"match": False, "confidence": 0.0, "similarities": {}}

def main():
    
    import librosa
    from colorama import Fore, Style
    
    # with open(classes_path, "r") as f:
    #     s = f.read()
    # num_classes = len(s.split("/n"))
    
    base_dir = "./"
    device = torch.device('cpu')
    
    model = ResNetMelLite.load_from_checkpoint(checkpoint_path, num_classes=123).to('cpu')
    
    # 269 class version
    # model = ResNetMelLite.load_from_checkpoint(checkpoint_path, num_classes=269).to('cpu')
    # self.model = ResNetMel.load_from_checkpoint(checkpoint_path, num_classes=387).to('cpu')
    model.model.fc[4] = torch.nn.Sequential()
    model.eval()
    
    # Initialize matcher
    # matcher = EnhancedSimilarityMatcher(positive_embeddings, negative_embeddings)
    print("ya here")
    # matcher = Matcher()
    
    positive_embeddings = []
    
    
    
    name = input(f"{Fore.BLUE}Enter the wakeword name you want to use{Style.RESET_ALL}({Fore.GREEN}shiva, shivan, shambu, alexa, sam, munez, nigga{Style.RESET_ALL}): ")
    if name == "alexa":
        audio_file_paths = fetch_audios(name)[:3]
    else:
        audio_file_paths = fetch_audios(name)
    print(f"{Fore.GREEN}Audio files fetched are: {audio_file_paths}{Style.RESET_ALL}")
    
    # audio_paths = [
    #     os.path.join("./Audios4testing/shiva_1.wav"),
    #     os.path.join("./Audios4testing/shiva_2.wav"),
    #     os.path.join("./Audios4testing/shiva_3.wav"),
    #     # os.path.join("./Audios4testing/alexa_4.wav"),
    #     # os.path.join("./Audios4testing/alexa_5.wav")
    # ]
    for path in audio_file_paths:
        mel_spec = load_and_preprocess_audio_file(path, max_duration=1.0)
        mel_spec_tensor = torch.tensor([mel_spec], dtype=torch.float32)
        mel_spec_tensor = mel_spec_tensor.unsqueeze(1)
        embs = model(mel_spec_tensor)
        # print(embs.dtype)
        positive_embeddings.append(embs)
        
    
    negative_embeddings = []
    audio_paths = [
        os.path.join("./Audios4testing/Thunderbolt_en-AU-jimm.mp3"),
        os.path.join("./Audios4testing/Thunderbolt_en-AU-kylie.mp3"),
        os.path.join("./Audios4testing/Skywalker_en-AU-jimm.mp3"),
        os.path.join("./Audios4testing/Skywalker_en-AU-kylie.mp3"),
        os.path.join("./Audios4testing/Hello0.mp3"),
        os.path.join("./Audios4testing/Hello1.mp3"),
        # os.path.join("./Audios4testing/Augh.wav"),
        # os.path.join("./Audios4testing/Augh2.wav"),
        # os.path.join("./Audios4testing/Ummmm.wav"),
        # os.path.join("./Audios4testing/Aahuhuhaah.wav"),
        # os.path.join("./Audios4testing/blah.wav"),
        # os.path.join("./Audios4testing/tap.wav"),
        # os.path.join("./Audios4testing/taptaptap.wav"),
    ]
    for path in audio_paths:
        mel_spec = load_and_preprocess_audio_file(path, max_duration=1.0)
        mel_spec_tensor = torch.tensor([mel_spec], dtype=torch.float32)
        mel_spec_tensor = mel_spec_tensor.unsqueeze(1)
        embs = model(mel_spec_tensor)
        negative_embeddings.append(embs)
        
        
    matcher = EnhancedSimilarityMatcher(positive_embeddings, negative_embeddings)
    # Initialize detector with your ONNX model path
    wake_word_detector = HotwordDetector(
        hotword=name,
        # reference_file="path_to_reference.json",  # Contains reference embeddings
        # reference_file="Shambu_23thModel.json",
        # reference_file="Alexa_23thModel.json",
        reference_file=f"./references/Shambu_27thModel_epoch9.json",
        # reference_file="Munez_25th_Model.json",
        # model_path="./resnet_50_arc/slim_93%_accuracy_72.7390%.onnx",
        model_path="ResnetMel",
        matcher=matcher,
        window_length=1.0,
        threshold=0.67  # Adjust based on your needs
        # threshold=0.79239375
    )
    
    print("no yay here")
    # Start microphone stream
    mic_stream = SimpleMicStream()
    mic_stream.start_stream()
    
    print(f"{Fore.YELLOW}Listening for wake word{Style.RESET_ALL} {Fore.GREEN}'{wake_word_detector.hotword}'{Style.RESET_ALL}...")
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