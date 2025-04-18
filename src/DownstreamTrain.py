import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Tuple, List, Dict, Optional
import random

# Define the base ResNetMel model (assumed to be available)
class ResNetMel(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        # This is a placeholder for your actual ResNetMel implementation
        # In practice, you would import your existing model here
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.embedding_size = 256
        self.fc = nn.Linear(self.embedding_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        # self.embedding is the representation before the classifier
        self.embedding = x
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def get_embedding(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return x


# Contrastive Loss Implementation
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        # label=1 for similar pairs, label=0 for dissimilar pairs
        distance = F.pairwise_distance(embedding1, embedding2)
        # For similar pairs, minimize distance; for dissimilar pairs, push distance beyond margin
        loss = label * distance.pow(2) + (1 - label) * F.relu(self.margin - distance).pow(2)
        return loss.mean()


# Triplet Loss Implementation
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        # Loss is positive when the negative should be pushed farther
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


# Audio processing utilities
class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
    
    def process(self, audio_path, duration=3.0):
        # Load audio file
        waveform, sr = torchaudio.load(audio_path)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Trim or pad to a fixed duration
        target_length = int(duration * self.sample_rate)
        if waveform.shape[1] < target_length:
            # Pad with zeros if audio is shorter
            padding = target_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        else:
            # Trim if audio is longer
            waveform = waveform[:, :target_length]
        
        # Convert to mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        # Apply log transformation
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Normalize
        mean = mel_spec.mean()
        std = mel_spec.std()
        mel_spec = (mel_spec - mean) / (std + 1e-9)
        
        # Add channel dimension for CNN input (1, n_mels, time)
        return mel_spec.unsqueeze(0)


# Dataset for Contrastive Learning (pairs of samples)
class ContrastiveAudioDataset(Dataset):
    def __init__(self, data_csv, audio_dir, processor, transform=None):
        """
        Args:
            data_csv: Path to CSV file with columns:
                - audio_path1: path to first audio file
                - audio_path2: path to second audio file
                - label: 1 if same class/similar, 0 if different class/dissimilar
            audio_dir: Directory where audio files are stored
            processor: AudioProcessor instance
            transform: Optional transform to be applied on a sample
        """
        self.data = pd.read_csv(data_csv)
        self.audio_dir = Path(audio_dir)
        self.processor = processor
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get paths and label
        path1 = self.audio_dir / row['audio_path1']
        path2 = self.audio_dir / row['audio_path2']
        label = float(row['label'])  # 1 for similar, 0 for dissimilar
        
        # Process audio files
        mel_spec1 = self.processor.process(path1)
        mel_spec2 = self.processor.process(path2)
        
        # Apply transforms if any
        if self.transform:
            mel_spec1 = self.transform(mel_spec1)
            mel_spec2 = self.transform(mel_spec2)
        
        return {'mel_spec1': mel_spec1, 'mel_spec2': mel_spec2, 'label': torch.tensor(label)}


# Dataset for Triplet Learning (anchor, positive, negative samples)
class TripletAudioDataset(Dataset):
    def __init__(self, data_csv, audio_dir, processor, transform=None):
        """
        Args:
            data_csv: Path to CSV file with columns:
                - anchor_path: path to anchor audio file
                - positive_path: path to positive audio file (same class as anchor)
                - negative_path: path to negative audio file (different class from anchor)
            audio_dir: Directory where audio files are stored
            processor: AudioProcessor instance
            transform: Optional transform to be applied on a sample
        """
        self.data = pd.read_csv(data_csv)
        self.audio_dir = Path(audio_dir)
        self.processor = processor
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get paths
        anchor_path = self.audio_dir / row['anchor_path']
        positive_path = self.audio_dir / row['positive_path']
        negative_path = self.audio_dir / row['negative_path']
        
        # Process audio files
        anchor_spec = self.processor.process(anchor_path)
        positive_spec = self.processor.process(positive_path)
        negative_spec = self.processor.process(negative_path)
        
        # Apply transforms if any
        if self.transform:
            anchor_spec = self.transform(anchor_spec)
            positive_spec = self.transform(positive_spec)
            negative_spec = self.transform(negative_spec)
        
        return {
            'anchor': anchor_spec, 
            'positive': positive_spec, 
            'negative': negative_spec
        }


# PyTorch Lightning Model
class AudioSimilarityModel(pl.LightningModule):
    def __init__(
        self, 
        base_model=None,
        embedding_size=256,
        contrastive_weight=0.5, 
        triplet_weight=0.5,
        contrastive_margin=1.0,
        triplet_margin=1.0,
        lr=0.0001
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['base_model'])
        
        # Use provided model or create a new one
        if base_model is None:
            self.base_model = ResNetMel(num_classes=10, dropout=0.2)
        else:
            self.base_model = base_model
        
        # Remove the classifier head
        self.embedding_size = embedding_size
        
        # Define loss functions
        self.contrastive_loss_fn = ContrastiveLoss(margin=contrastive_margin)
        self.triplet_loss_fn = TripletLoss(margin=triplet_margin)
        
        # Loss weights
        self.contrastive_weight = contrastive_weight
        self.triplet_weight = triplet_weight
        self.lr = lr
    
    def forward(self, x):
        # Get embeddings from the base model
        return self.base_model.get_embedding(x)
    
    def contrastive_step(self, batch):
        # Unpack batch
        mel_spec1, mel_spec2, labels = batch['mel_spec1'], batch['mel_spec2'], batch['label']
        
        # Get embeddings
        embedding1 = self(mel_spec1)
        embedding2 = self(mel_spec2)
        
        # Calculate contrastive loss
        loss = self.contrastive_loss_fn(embedding1, embedding2, labels)
        
        return loss
    
    def triplet_step(self, batch):
        # Unpack batch
        anchor, positive, negative = batch['anchor'], batch['positive'], batch['negative']
        
        # Get embeddings
        anchor_embedding = self(anchor)
        positive_embedding = self(positive)
        negative_embedding = self(negative)
        
        # Calculate triplet loss
        loss = self.triplet_loss_fn(anchor_embedding, positive_embedding, negative_embedding)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        # Determine which loss to use based on batch structure
        if 'mel_spec1' in batch and 'mel_spec2' in batch and 'label' in batch:
            # Contrastive loss
            loss = self.contrastive_step(batch)
            self.log('train_contrastive_loss', loss)
            return self.contrastive_weight * loss
        elif 'anchor' in batch and 'positive' in batch and 'negative' in batch:
            # Triplet loss
            loss = self.triplet_step(batch)
            self.log('train_triplet_loss', loss)
            return self.triplet_weight * loss
        else:
            # If batch contains both types (implemented as a dictionary with both structures)
            contrastive_loss = self.contrastive_step(batch['contrastive'])
            triplet_loss = self.triplet_step(batch['triplet'])
            
            self.log('train_contrastive_loss', contrastive_loss)
            self.log('train_triplet_loss', triplet_loss)
            
            total_loss = self.contrastive_weight * contrastive_loss + self.triplet_weight * triplet_loss
            self.log('train_loss', total_loss)
            
            return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Similar structure to training_step
        if 'mel_spec1' in batch and 'mel_spec2' in batch and 'label' in batch:
            loss = self.contrastive_step(batch)
            self.log('val_contrastive_loss', loss)
            return loss
        elif 'anchor' in batch and 'positive' in batch and 'negative' in batch:
            loss = self.triplet_step(batch)
            self.log('val_triplet_loss', loss)
            return loss
        else:
            contrastive_loss = self.contrastive_step(batch['contrastive'])
            triplet_loss = self.triplet_step(batch['triplet'])
            
            self.log('val_contrastive_loss', contrastive_loss)
            self.log('val_triplet_loss', triplet_loss)
            
            total_loss = self.contrastive_weight * contrastive_loss + self.triplet_weight * triplet_loss
            self.log('val_loss', total_loss)
            
            return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


# Data module to handle both contrastive and triplet datasets
class AudioSimilarityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        contrastive_train_csv=None,
        contrastive_val_csv=None,
        triplet_train_csv=None,
        triplet_val_csv=None,
        audio_dir='./data/audio',
        batch_size=32,
        num_workers=4,
        sample_rate=16000,
        n_mels=128,
        duration=3.0
    ):
        super().__init__()
        self.contrastive_train_csv = contrastive_train_csv
        self.contrastive_val_csv = contrastive_val_csv
        self.triplet_train_csv = triplet_train_csv
        self.triplet_val_csv = triplet_val_csv
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create audio processor
        self.processor = AudioProcessor(
            sample_rate=sample_rate,
            n_mels=n_mels
        )
        
        # Dataset attributes
        self.contrastive_train_dataset = None
        self.contrastive_val_dataset = None
        self.triplet_train_dataset = None
        self.triplet_val_dataset = None
    
    def setup(self, stage=None):
        # Create datasets
        if self.contrastive_train_csv:
            self.contrastive_train_dataset = ContrastiveAudioDataset(
                self.contrastive_train_csv, self.audio_dir, self.processor
            )
        
        if self.contrastive_val_csv:
            self.contrastive_val_dataset = ContrastiveAudioDataset(
                self.contrastive_val_csv, self.audio_dir, self.processor
            )
        
        if self.triplet_train_csv:
            self.triplet_train_dataset = TripletAudioDataset(
                self.triplet_train_csv, self.audio_dir, self.processor
            )
        
        if self.triplet_val_csv:
            self.triplet_val_dataset = TripletAudioDataset(
                self.triplet_val_csv, self.audio_dir, self.processor
            )
    
    def train_dataloader(self):
        loaders = []
        
        if self.contrastive_train_dataset:
            loaders.append(DataLoader(
                self.contrastive_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            ))
        
        if self.triplet_train_dataset:
            loaders.append(DataLoader(
                self.triplet_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            ))
        
        # If both datasets are available, alternate between them
        if len(loaders) > 1:
            # Return a combined loader that will cycle through both types
            # In practice, you might need to implement a custom DataLoader that alternates
            # between contrastive and triplet batches
            return loaders[0]  # For now, return just the first one
        elif len(loaders) == 1:
            return loaders[0]
        else:
            raise ValueError("No training datasets are available")
    
    def val_dataloader(self):
        loaders = []
        
        if self.contrastive_val_dataset:
            loaders.append(DataLoader(
                self.contrastive_val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            ))
        
        if self.triplet_val_dataset:
            loaders.append(DataLoader(
                self.triplet_val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            ))
        
        # If both datasets are available, return a list of loaders
        if len(loaders) > 1:
            return loaders[0]  # For now, return just the first one
        elif len(loaders) == 1:
            return loaders[0]
        else:
            raise ValueError("No validation datasets are available")


# Example usage
def train_similarity_model():
    # Create the model
    base_model = ResNetMel(num_classes=10, dropout=0.2)
    model = AudioSimilarityModel(
        base_model=base_model,
        embedding_size=256,
        contrastive_weight=0.5,
        triplet_weight=0.5,
        contrastive_margin=1.0,
        triplet_margin=1.0,
        lr=0.0001
    )
    
    # Create data module
    data_module = AudioSimilarityDataModule(
        contrastive_train_csv='data/contrastive_train.csv',
        contrastive_val_csv='data/contrastive_val.csv',
        triplet_train_csv='data/triplet_train.csv',
        triplet_val_csv='data/triplet_val.csv',
        audio_dir='data/audio',
        batch_size=32,
        num_workers=4
    )
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        filename='similarity-{epoch:02d}-{val_loss:.4f}',
        save_weights_only=True
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        num_nodes=1,
        enable_checkpointing=True
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Save the final model
    trainer.save_checkpoint("final_audio_similarity_model.ckpt")


if __name__ == "__main__":
    train_similarity_model()