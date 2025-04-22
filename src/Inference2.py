import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
from tqdm import tqdm
import pandas as pd
from Preprocessing import load_and_preprocess_audio_file
import numpy as np
from Utils import  Matcher
import librosa

audio, sr = librosa.load("./Augmented/output_aug1.wav")
print(len(audio), sr)

audio, sr = librosa.load("output.wav")
print(len(audio), sr)
exit(0)

class ResNetMel(pl.LightningModule):
    def __init__(self, mel_bins=40, time_frames=100, num_classes=10, dropout=0.0, learning_rate=0.001, pretrained=True):
        """
        ResNet model modified to work with mel spectrograms (1 channel input)
        
        Args:
            mel_bins (int): Number of mel frequency bins
            time_frames (int): Number of time frames
            num_classes (int): Number of output classes
            dropout (float): Dropout rate
            learning_rate (float): Learning rate for optimizer
            pretrained (bool): Whether to use pretrained weights for layers beyond the first conv
        """
        super(ResNetMel, self).__init__()
        self.lr = learning_rate
        self.num_classes = num_classes
        self.mel_bins = mel_bins
        
        # Load the standard ResNet18 model
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        
        
        # Modify the first convolutional layer to accept 1 channel instead of 3
        # We'll create a new conv1 with 1 input channel but same output channels
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            1, 
            original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=(original_conv1.bias is not None)
        )
        
        # If using pretrained weights, we need to adapt the weights for the first layer
        if pretrained:
            # Average the weights across the 3 input channels to create weights for 1 channel
            new_weights = original_conv1.weight.data.mean(dim=1, keepdim=True)
            self.model.conv1.weight.data = new_weights
        
        # Modify the final fully connected layer
        # self.model.fc = nn.Sequential(
        #     nn.Linear(512, 4096),
        #     nn.ReLU(),
        #     # nn.Dropout(dropout),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(),
        #     nn.Linear(4096, num_classes)
        # )
        
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        
        # Loss function for training
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input mel spectrogram of shape [batch_size, 1, mel_bins, time_frames]
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Tuple of (inputs, labels)
        """
        audio, labels = batch
        
        # Forward pass
        y_pred = self(audio)
        
        # Calculate loss
        loss = self.loss(y_pred, labels)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        
        # Calculate accuracy
        preds = torch.argmax(y_pred, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        
        Args:
            batch: Tuple of (inputs, labels)
        """
        audio, labels = batch
        
        # Forward pass
        y_preds = self(audio)
        
        # Calculate loss
        loss = self.loss(y_preds, labels)
        
        # Calculate accuracy
        preds = torch.argmax(y_preds, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
# with open("./mswc_cache/classes.txt", 'r') as f:
#     s = f.read()
   
with open("./dataset_cache2/classes.txt", 'r') as f:
    s = f.read()

classes = s.split("\n")
num_classes = len(classes)


# Wakeword only Dataset
checkpoint_path = "./lightning_logs/version_17/checkpoints/epoch=24-step=61375.ckpt"
# checkpoint_path = "./lightning_logs/version_18/checkpoints/epoch=4-step=12275.ckpt"

# MSWC Dataset Model
# checkpoint_path = "./lightning_logs/version_21/checkpoints/epoch=5-step=18624.ckpt"
# checkpoint_path = "./lightning_logs/version_22/checkpoints/epoch=49-step=155200.ckpt"

# checkpoint_path = "./lightning_logs/version_23/checkpoints/epoch=14-step=46560.ckpt"
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
# model.model.fc[4] = nn.Sequential()
model.model.fc[1] = nn.Sequential()

print(model.named_modules)

out = model(mel_spec_tensor)
print(out.shape)

out1 = model(mel_spec_tensor1)
print(out1.shape)

matcher = Matcher()
print(torch.cosine_similarity(out, out1, dim=-1))
print(matcher.match(out, out1))