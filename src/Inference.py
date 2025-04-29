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
from Trainer import ResNetMel
from Utils import EnhancedSimilarityMatcher, Matcher
    
with open("./mswc_cache/classes.txt", 'r') as f:
    s = f.read()
   
# with open("./AI_audios_cache/classes.txt", 'r') as f:
#     s = f.read()
classes = s.split("\n")
num_classes = len(classes)

checkpoint_path = "./lightning_logs/version_23/checkpoints/epoch=14-step=46560.ckpt"
# checkpoint_path = "./lightning_logs/version_25/checkpoints/epoch=13-step=44142.ckpt"
# checkpoint_path = "./lightning_logs/version_26/checkpoints/epoch=24-step=78825.ckpt"
# Wakeword only Dataset
model = ResNetMel.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
model.eval()

mel_spec = load_and_preprocess_audio_file("./Audios4testing/munez_2.wav", max_duration=1.0)
mel_spec_tensor = torch.tensor([mel_spec], dtype=torch.float32)
mel_spec_tensor = mel_spec_tensor.unsqueeze(1)
mel_spec_tensor = mel_spec_tensor.cuda()
print(mel_spec_tensor.shape)
out = model(mel_spec_tensor)
index = torch.argmax(out, dim=-1)
print(out.shape, index)
print(classes[index])

mel_spec1 = load_and_preprocess_audio_file("./Audios4testing/munez_3.wav", max_duration=1.0)

mel_spec_tensor1 = torch.tensor([mel_spec1], dtype=torch.float32)
mel_spec_tensor1 = mel_spec_tensor1.unsqueeze(1)
mel_spec_tensor1 = mel_spec_tensor1.cuda()
print(mel_spec_tensor1.shape)
out = model(mel_spec_tensor1)
index = torch.argmax(out, dim=-1)
print(out.shape, index)
print(classes[index])
# model.model.fc[1] = nn.Sequential()

model.model.fc[4] = nn.Sequential()
print(model.named_modules)

out = model(mel_spec_tensor)
print(out.shape)

out1 = model(mel_spec_tensor1)
print(out1.shape)

matcher = Matcher()
print(torch.cosine_similarity(out, out1, dim=-1))
print(matcher.match(out, out1))
# import json
# # Save the embeddings to a JSON file
# data = {"embeddings": out.cpu().detach().numpy().tolist()}
# with open("path_to_reference.json", "w") as f:
#     f.write(json.dumps(data))
# print("saved embeddings")