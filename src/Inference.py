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

# with open("./dataset_cache/classes.txt", 'r') as f:
#     s = f.read()
    
with open("./mswc_cache/classes.txt", 'r') as f:
    s = f.read()
   
# max_duration = torch.load("./mswc_cache/max_duration.pt")
# max_length = torch.load("./mswc_cache/max_length.pt")

# print(max_duration)
# print(max_length)

# exit(0)
classes = s.split("\n")
num_classes = len(classes)
# CRNN
# checkpoint_path = "./lightning_logs/version_2/checkpoints/epoch=9-step=130.ckpt"
# checkpoint_path = "./lightning_logs/version_1/checkpoints/epoch=19-step=260.ckpt"
# checkpoint_path = "./lightning_logs/version_0/checkpoints/epoch=29-step=390.ckpt"
# checkpoint_path = "./lightning_logs/version_5/checkpoints/epoch=29-step=780.ckpt"
# checkpoint_path = "./lightning_logs/version_9/checkpoints/epoch=19-step=48580.ckpt"

# checkpoint_path = "./lightning_logs/version_10/checkpoints/epoch=49-step=121450.ckpt"
# checkpoint_path = "./lightning_logs/version_11/checkpoints/epoch=79-step=194320.ckpt"
# Resnet18
# checkpoint_path = "./lightning_logs/version_14/checkpoints/epoch=18-step=46151.ckpt"
# checkpoint_path = "./lightning_logs/version_15/checkpoints/epoch=14-step=36435.ckpt"
# checkpoint_path = "./lightning_logs/version_16/checkpoints/epoch=7-step=19432.ckpt"

# Wakeword only Dataset
# checkpoint_path = "./lightning_logs/version_17/checkpoints/epoch=24-step=61375.ckpt"
# checkpoint_path = "./lightning_logs/version_18/checkpoints/epoch=4-step=12275.ckpt"

# MSWC Dataset Model
# checkpoint_path = "./lightning_logs/version_21/checkpoints/epoch=5-step=18624.ckpt"

checkpoint_path = "./lightning_logs/version_22/checkpoints/epoch=49-step=155200.ckpt"
# model = CRNN.load_from_checkpoint(checkpoint_path, num_classes=74)
# model = CRNN.load_from_checkpoint(checkpoint_path, num_classes=143)
# model = CRNN.load_from_checkpoint(checkpoint_path, num_classes=175)
# model = ResNetMel.load_from_checkpoint(checkpoint_path, num_classes=175)
# Wakeword only Dataset
model = ResNetMel.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
model.eval()
# lightning_logs\version_2\checkpoints\epoch=9-step=130.ckpt
# model = model.load_from_checkpoint("./lightning_logs/version_2/checkpoints/epoch=9-step=130.ckpt")
# print(model.named_modules)

# mel_spec = load_and_preprocess_audio_file("./Audio_dataset/Aneesh/Aneesh_d088cdf6-0ef0-4656-aea8-eb9b004e82eb_hi.wav", max_duration=4.14)
# mel_spec = load_and_preprocess_audio_file("./Audio_dataset/Hello/Hello_shouted_2.wav", max_duration=5.63)
# mel_spec = load_and_preprocess_audio_file("./Audio_dataset/Bolo/Bolo_bec003e2-3cb3-429c-8468-206a393c67ad_hi.wav", max_duration=5.63)
# mel_spec = load_and_preprocess_audio_file("Nobita.wav", max_duration=4.14)
# mel_spec = load_and_preprocess_audio_file("Nobita.wav", max_duration=1.0)
mel_spec = load_and_preprocess_audio_file("./Audio_dataset/Bolo/Bolo_bec003e2-3cb3-429c-8468-206a393c67ad_hi.wav", max_duration=1.0)
# mel_spec = load_and_preprocess_audio_file("./Audio_dataset/Eli/Eli_57c63422-d911-4666-815b-0c332e4d7d6a_en.wav", max_duration=1.0)
mel_spec_tensor = torch.tensor([mel_spec], dtype=torch.float32)
mel_spec_tensor = mel_spec_tensor.unsqueeze(1)
mel_spec_tensor = mel_spec_tensor.cuda()
print(mel_spec_tensor.shape)
out = model(mel_spec_tensor)
index = torch.argmax(out, dim=-1)
print(out.shape, index)
print(classes[index])

# mel_spec1 = load_and_preprocess_audio_file("./Audio_dataset/Jonathan/Jonathan_00967b2f-88a6-4a31-8153-110a92134b9f_en.wav", max_duration=4.14)
# mel_spec1 = load_and_preprocess_audio_file("AC Chalu Karo.wav", max_duration=5.63)
# mel_spec1 = load_and_preprocess_audio_file("Bolo.wav", max_duration=4.14)
# mel_spec1 = load_and_preprocess_audio_file("Bolo.wav", max_duration=1.0)
mel_spec1 = load_and_preprocess_audio_file("./Audio_dataset/Bolo/Bolo_bec003e2-3cb3-429c-8468-206a393c67ad_hi.wav", max_duration=1.0)
# mel_spec1 = load_and_preprocess_audio_file("Nobi.wav", max_duration=1.0)
mel_spec_tensor1 = torch.tensor([mel_spec1], dtype=torch.float32)
mel_spec_tensor1 = mel_spec_tensor1.unsqueeze(1)
mel_spec_tensor1 = mel_spec_tensor1.cuda()
print(mel_spec_tensor1.shape)
out = model(mel_spec_tensor1)
index = torch.argmax(out, dim=-1)
print(out.shape, index)
print(classes[index])
model.model.fc[1] = nn.Sequential()
print(model.named_modules)

out = model(mel_spec_tensor)
print(out.shape)

out1 = model(mel_spec_tensor1)
print(out1.shape)

matcher = Matcher()
print(torch.cosine_similarity(out, out1, dim=-1))
print(matcher.match(out, out1))