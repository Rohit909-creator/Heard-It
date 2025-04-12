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


with open("./dataset_cache/classes.txt", 'r') as f:
    s = f.read()
    
classes = s.split("\n")

model = CRNN(num_classes=74)
model.eval()
# lightning_logs\version_2\checkpoints\epoch=9-step=130.ckpt
model = model.load_from_checkpoint("./lightning_logs/version_2/checkpoints/epoch=9-step=130.ckpt")
print(model.named_modules)

mel_spec = load_and_preprocess_audio_file("Hello_9b953e7b-86a8-42f0-b625-1434fb15392b_hi.wav")

mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32)
mel_spec_tensor = mel_spec_tensor.unsqueeze(1)
out = model(mel_spec_tensor)
index = torch.argmax(out, dim=-1)
print(out.shape, index)
print(classes[index])
model.linear_map = nn.Sequential()
print(model.named_modules)



