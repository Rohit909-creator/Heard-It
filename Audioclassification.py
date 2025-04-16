import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from torchaudio.sox_effects import apply_effects_file
import librosa
effects = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]
def map_to_array(example):
    speech, _ = apply_effects_file(example["file"], effects)
    example["speech"] = speech.squeeze(0).numpy()
    return example

# load a demo dataset and read audio files
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ks")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks")
waveform, sr = librosa.load('Bolo.wav')
# compute attention masks and normalize the waveform if needed
inputs = feature_extractor(waveform, sampling_rate=16000, padding=True, return_tensors="pt")

logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]
