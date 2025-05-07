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
from Trainer import ResNetMel, ResNetMelLite
from Utils import EnhancedSimilarityMatcher, Matcher


# More efficient implementation for high-dimensional feature vectors
def efficient_dtw(x, y, radius=None):
    """
    Efficient DTW implementation for high-dimensional feature vectors.
    
    Parameters:
    - x, y: PyTorch tensors of shape [batch_size, feature_dim] or [feature_dim]
    - radius: Sakoe-Chiba band radius (window size)
    
    Returns:
    - distance: DTW distance between the vectors
    """
    # Convert to numpy for faster computation on CPU if needed
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
        
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = y
    
    # Ensure proper dimensions
    if len(x_np.shape) == 2 and x_np.shape[0] == 1:
        x_np = x_np.squeeze(0)
    if len(y_np.shape) == 2 and y_np.shape[0] == 1:
        y_np = y_np.squeeze(0)
    
    # If inputs are single high-dimensional vectors, return Euclidean distance
    if len(x_np.shape) == 1 or (len(x_np.shape) == 2 and min(x_np.shape) == 1):
        return np.linalg.norm(x_np - y_np)
    
    n, m = len(x_np), len(y_np)
    
    # Use radius for windowing to improve efficiency
    if radius is None:
        radius = max(n, m)
    
    # Initialize cost matrix
    D = np.full((n+1, m+1), np.inf)
    D[0, 0] = 0
    
    # Fill the cost matrix with windowing
    for i in range(1, n+1):
        j_start = max(1, i-radius)
        j_end = min(m+1, i+radius+1)
        
        for j in range(j_start, j_end):
            # Calculate distance between points
            cost = np.linalg.norm(x_np[i-1] - y_np[j-1])
            
            # Find minimum cost path
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    
    # Return the final DTW distance
    return D[n, m]

with open("./mswc_cache/classes.txt", 'r') as f:
    s = f.read()
   
# with open("./AI_audios_cache/classes.txt", 'r') as f:
#     s = f.read()
classes = s.split("\n")
num_classes = len(classes)
# checkpoint_path = "./lightning_logs/version_23/checkpoints/epoch=14-step=46560.ckpt"
checkpoint_path = "./lightning_logs/version_27/checkpoints/epoch=9-step=31040.ckpt"
# checkpoint_path = "./lightning_logs/version_25/checkpoints/epoch=13-step=44142.ckpt"
# checkpoint_path = "./lightning_logs/version_26/checkpoints/epoch=24-step=78825.ckpt"
# Wakeword only Dataset
model = ResNetMelLite.load_from_checkpoint(checkpoint_path, num_classes=num_classes)

model.eval()

mel_spec = load_and_preprocess_audio_file("./Audios4testing/sam_1.wav", max_duration=1.0)
mel_spec_tensor = torch.tensor([mel_spec], dtype=torch.float32)
mel_spec_tensor = mel_spec_tensor.unsqueeze(1)
mel_spec_tensor = mel_spec_tensor.cuda()
print(mel_spec_tensor.shape)
out = model(mel_spec_tensor)
index = torch.argmax(out, dim=-1)
print(out.shape, index)
print(classes[index])

mel_spec1 = load_and_preprocess_audio_file("./Audios4testing/sam_2.wav", max_duration=1.0)

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

# print(out1[0].shape)


distance = efficient_dtw(out, out1)

print(f"DTW distance between TSLA and AMZN: {distance}")


# import json
# # Save the embeddings to a JSON file
# data = {"embeddings": out.cpu().detach().numpy().tolist()}
# with open("path_to_reference.json", "w") as f:
#     f.write(json.dumps(data))
# print("saved embeddings")