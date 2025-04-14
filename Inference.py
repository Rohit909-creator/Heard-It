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
from Trainer import CRNN
from Utils import EnhancedSimilarityMatcher, Matcher

with open("./dataset_cache/classes.txt", 'r') as f:
    s = f.read()
    
classes = s.split("\n")
# checkpoint_path = "./lightning_logs/version_2/checkpoints/epoch=9-step=130.ckpt"
# checkpoint_path = "./lightning_logs/version_1/checkpoints/epoch=19-step=260.ckpt"
# checkpoint_path = "./lightning_logs/version_0/checkpoints/epoch=29-step=390.ckpt"

checkpoint_path = "./lightning_logs/version_5/checkpoints/epoch=29-step=780.ckpt"
# model = CRNN.load_from_checkpoint(checkpoint_path, num_classes=74)
model = CRNN.load_from_checkpoint(checkpoint_path, num_classes=143)
model.eval()
# lightning_logs\version_2\checkpoints\epoch=9-step=130.ckpt
# model = model.load_from_checkpoint("./lightning_logs/version_2/checkpoints/epoch=9-step=130.ckpt")
print(model.named_modules)

# mel_spec = load_and_preprocess_audio_file("./Audio_dataset/AC chalu karo/AC chalu karo_9b953e7b-86a8-42f0-b625-1434fb15392b_hi.wav", max_duration=5.63)
mel_spec = load_and_preprocess_audio_file("Bolo.wav", max_duration=5.63)
# mel_spec = load_and_preprocess_audio_file("./Audio_dataset/Bolo/Bolo_bec003e2-3cb3-429c-8468-206a393c67ad_hi.wav", max_duration=5.63)
mel_spec_tensor = torch.tensor([mel_spec], dtype=torch.float32)
mel_spec_tensor = mel_spec_tensor.unsqueeze(1)
mel_spec_tensor = mel_spec_tensor.cuda()
print(mel_spec_tensor.shape)
out = model(mel_spec_tensor)
index = torch.argmax(out, dim=-1)
print(out.shape, index)
print(classes[index])

# mel_spec1 = load_and_preprocess_audio_file("./Audio_dataset/Aapka naam kya hai/Aapka naam kya hai_9b953e7b-86a8-42f0-b625-1434fb15392b_hi.wav", max_duration=5.63)
mel_spec1 = load_and_preprocess_audio_file("Bolo.wav", max_duration=5.63)
mel_spec_tensor1 = torch.tensor([mel_spec1], dtype=torch.float32)
mel_spec_tensor1 = mel_spec_tensor1.unsqueeze(1)
mel_spec_tensor1 = mel_spec_tensor1.cuda()
print(mel_spec_tensor1.shape)
out = model(mel_spec_tensor1)
index = torch.argmax(out, dim=-1)
print(out.shape, index)
print(classes[index])
model.linear_map = nn.Sequential()
print(model.named_modules)

out = model(mel_spec_tensor)
print(out.shape)

out1 = model(mel_spec_tensor1)
print(out1.shape)

matcher = Matcher()
print(torch.cosine_similarity(out, out1, dim=-1))
print(matcher.match(out, out1))