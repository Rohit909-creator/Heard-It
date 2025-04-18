import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
from tqdm import tqdm
import pandas as pd
from Preprocessing import load_and_preprocess_audio_file
import numpy as np
from Trainer import CRNN, ResNetMel
from Utils import EnhancedSimilarityMatcher, Matcher
    
with open("./mswc_cache/classes.txt", 'r') as f:
    s = f.read()
   
# max_duration = torch.load("./mswc_cache/max_duration.pt")
# max_length = torch.load("./mswc_cache/max_length.pt")

# print(max_duration)
# print(max_length)

# exit(0)
classes = s.split("\n")
num_classes = len(classes)


# MSWC Dataset Model
# checkpoint_path = "./lightning_logs/version_21/checkpoints/epoch=5-step=18624.ckpt"
# checkpoint_path = "./lightning_logs/version_22/checkpoints/epoch=49-step=155200.ckpt"

checkpoint_path = "./lightning_logs/version_23/checkpoints/epoch=14-step=46560.ckpt"
# Wakeword only Dataset
model = ResNetMel.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
model.eval()
# lightning_logs\version_2\checkpoints\epoch=9-step=130.ckpt
# model = model.load_from_checkpoint("./lightning_logs/version_2/checkpoints/epoch=9-step=130.ckpt")
# print(model.named_modules)

# mel_spec = load_and_preprocess_audio_file("Nobita.wav", max_duration=1.0)
# mel_spec = load_and_preprocess_audio_file("./Audio_dataset/Bolo/Bolo_bec003e2-3cb3-429c-8468-206a393c67ad_hi.wav", max_duration=1.0)
mel_spec = load_and_preprocess_audio_file("./Audio_dataset/Eli/Eli_57c63422-d911-4666-815b-0c332e4d7d6a_en.wav", max_duration=1.0)
mel_spec_tensor = torch.tensor([mel_spec], dtype=torch.float32)
mel_spec_tensor = mel_spec_tensor.unsqueeze(1)
mel_spec_tensor = mel_spec_tensor.cuda()
print(mel_spec_tensor.shape)
out = model(mel_spec_tensor)
index = torch.argmax(out, dim=-1)
print(out.shape, index)
print(classes[index])

# mel_spec1 = load_and_preprocess_audio_file("Bolo.wav", max_duration=1.0)
mel_spec1 = load_and_preprocess_audio_file("Nobi.wav", max_duration=1.0)
mel_spec_tensor1 = torch.tensor([mel_spec1], dtype=torch.float32)
mel_spec_tensor1 = mel_spec_tensor1.unsqueeze(1)
mel_spec_tensor1 = mel_spec_tensor1.cuda()
print(mel_spec_tensor1.shape)
out = model(mel_spec_tensor1)
index = torch.argmax(out, dim=-1)
print(out.shape, index)
print(classes[index])
model.model.fc[4] = nn.Sequential()
print(model.named_modules)

out = model(mel_spec_tensor)
print(out.shape)

out1 = model(mel_spec_tensor1)
print(out1.shape)

matcher = Matcher()
print(torch.cosine_similarity(out, out1, dim=-1))
print(matcher.match(out, out1))