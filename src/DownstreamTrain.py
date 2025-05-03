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
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle
import json
from Trainer import ResNetMel
from Preprocessing import load_and_preprocess_audio_file

def preprocess_audio_dataset(
    audio_dir="./Audio_dataset",
    contrastive_train_csv="data/contrastive_train.csv",
    contrastive_val_csv="data/contrastive_val.csv",
    cache_dir="./audio_cache",
    sample_rate=16000,
    n_mels=40,
    n_fft=1024,
    hop_length=512,
    max_duration=1.0,
    force_recompute=False
):
    """
    Preprocess audio dataset and save as cached tensors
    
    Args:
        audio_dir: Directory containing audio files
        contrastive_train_csv: CSV file with training pairs
        contrastive_val_csv: CSV file with validation pairs
        cache_dir: Directory to save cached tensors
        sample_rate: Audio sample rate
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length
        max_duration: Maximum audio duration in seconds
        force_recompute: Force recomputation of features even if cache exists
        
    Returns:
        Dictionary with paths to cached files
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    # Cache file paths
    cache_paths = {
        "train_specs1": cache_dir / "train_specs1.pt",
        "train_specs2": cache_dir / "train_specs2.pt",
        "train_labels": cache_dir / "train_labels.pt",
        "val_specs1": cache_dir / "val_specs1.pt",
        "val_specs2": cache_dir / "val_specs2.pt",
        "val_labels": cache_dir / "val_labels.pt",
        "metadata": cache_dir / "metadata.json"
    }
    
    # Check if cache exists
    if not force_recompute and all(path.exists() for path in cache_paths.values()):
        print("Using cached preprocessed data...")
        
        # Load metadata
        with open(cache_paths["metadata"], 'r') as f:
            metadata = json.load(f)
            
        return cache_paths, metadata
    
    print("Preprocessing audio files and creating cache...")
    
    # Create mel spectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Process audio function
    def process_audio_file(file_path):
        try:
            # Use default backend - works with most wav files
            waveform, sr = torchaudio.load(file_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            
            # Trim or pad to a fixed duration
            target_length = int(max_duration * sample_rate)
            if waveform.shape[1] < target_length:
                # Pad with zeros if audio is shorter
                padding = target_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            else:
                # Trim if audio is longer
                waveform = waveform[:, :target_length]
            
            # Convert to mel spectrogram
            mel_spec = mel_spectrogram(waveform)
            # Apply log transformation
            mel_spec = torch.log(mel_spec + 1e-9)
            
            # Normalize
            mean = mel_spec.mean()
            std = mel_spec.std()
            mel_spec = (mel_spec - mean) / (std + 1e-9)
            
            # Add channel dimension for CNN input (1, n_mels, time)
            return mel_spec.unsqueeze(0), True
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None, False
    
    # Process audio files and create cache
    
    # First, gather all unique file paths from both CSVs
    audio_files_dict = {}  # file_path -> processed tensor
    
    # Load training CSV
    train_df = pd.read_csv(contrastive_train_csv)
    # Load validation CSV
    val_df = pd.read_csv(contrastive_val_csv)
    
    # Get all unique file paths
    all_files = set()
    audio_dir = Path(audio_dir)
    
    for df in [train_df, val_df]:
        for col in ['audio_path1', 'audio_path2']:
            if col in df.columns:
                all_files.update(df[col].tolist())
    
    # Process all unique audio files
    print(f"Processing {len(all_files)} unique audio files...")
    for file_path in tqdm(all_files):
        full_path = audio_dir / file_path
        if full_path.exists():
            mel_spec, success = process_audio_file(full_path)
            if success:
                audio_files_dict[file_path] = mel_spec
        else:
            print(f"Warning: File not found: {full_path}")
    
    # Create tensor datasets for training
    print("Creating training tensors...")
    train_specs1 = []
    train_specs2 = []
    train_labels = []
    
    valid_rows = 0
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        if row['audio_path1'] in audio_files_dict and row['audio_path2'] in audio_files_dict:
            train_specs1.append(audio_files_dict[row['audio_path1']])
            train_specs2.append(audio_files_dict[row['audio_path2']])
            train_labels.append(float(row['label']))
            valid_rows += 1
    
    if valid_rows == 0:
        raise ValueError("No valid training pairs found! Check your data paths.")
    
    print(f"Found {valid_rows} valid training pairs out of {len(train_df)} total")
    
    # Stack tensors
    train_specs1 = torch.stack(train_specs1)
    train_specs2 = torch.stack(train_specs2)
    train_labels = torch.tensor(train_labels)
    
    # Create tensor datasets for validation
    print("Creating validation tensors...")
    val_specs1 = []
    val_specs2 = []
    val_labels = []
    
    valid_rows = 0
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        if row['audio_path1'] in audio_files_dict and row['audio_path2'] in audio_files_dict:
            val_specs1.append(audio_files_dict[row['audio_path1']])
            val_specs2.append(audio_files_dict[row['audio_path2']])
            val_labels.append(float(row['label']))
            valid_rows += 1
    
    if valid_rows == 0:
        raise ValueError("No valid validation pairs found! Check your data paths.")
    
    print(f"Found {valid_rows} valid validation pairs out of {len(val_df)} total")
    
    # Stack tensors
    val_specs1 = torch.stack(val_specs1)
    val_specs2 = torch.stack(val_specs2)
    val_labels = torch.tensor(val_labels)
    
    # Save to cache
    print("Saving to cache...")
    torch.save(train_specs1, cache_paths["train_specs1"])
    torch.save(train_specs2, cache_paths["train_specs2"])
    torch.save(train_labels, cache_paths["train_labels"])
    torch.save(val_specs1, cache_paths["val_specs1"])
    torch.save(val_specs2, cache_paths["val_specs2"])
    torch.save(val_labels, cache_paths["val_labels"])
    
    # Save metadata
    metadata = {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "max_duration": max_duration,
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "feature_dim": train_specs1.shape[1:] if len(train_specs1) > 0 else None
    }
    
    with open(cache_paths["metadata"], 'w') as f:
        json.dump(metadata, f)
    
    print("Preprocessing complete!")
    return cache_paths, metadata


# Define the base ResNetMel model
class ResNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        # This is a placeholder for your actual ResNetMel implementation
        # In practice, you would import your existing model here
        model = ResNetMel.load_from_checkpoint(checkpoint_path = "./lightning_logs/version_23/checkpoints/epoch=14-step=46560.ckpt", num_classes=52).to('cpu')
        model.model.fc[4] = nn.Sequential()
        model.train()
        self.actual_model = model
        
    def forward(self, x):
        # print(f"Shape: {x.shape}")
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
            # print(f"After modification: {x.shape}")
        x = self.actual_model(x)    
        self.embedding = x
        return x
    
    def get_embedding(self, x):
        x = self.forward(x)
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


# Dataset for Contrastive Learning with cached tensors
class CachedContrastiveDataset(Dataset):
    def __init__(self, specs1, specs2, labels, transform=None):
        """
        Args:
            specs1: Tensor of first spectrograms (N, C, H, W)
            specs2: Tensor of second spectrograms (N, C, H, W)
            labels: Tensor of labels (1 for similar, 0 for dissimilar)
            transform: Optional transform to be applied
        """
        self.specs1 = specs1
        self.specs2 = specs2
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        spec1 = self.specs1[idx]
        spec2 = self.specs2[idx]
        label = self.labels[idx]
        
        if self.transform:
            spec1 = self.transform(spec1)
            spec2 = self.transform(spec2)
        
        return {'mel_spec1': spec1, 'mel_spec2': spec2, 'label': label}


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
            self.base_model = ResNet(num_classes=52, dropout=0.7)
        else:
            self.base_model = base_model
        
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
    
    def training_step(self, batch, batch_idx):
        # Process contrastive batch
        loss = self.contrastive_step(batch)
        self.log('train_contrastive_loss', loss)
        # Also log the total loss as val_loss for checkpoint monitoring
        self.log('val_loss', loss)  # This is needed for model checkpointing
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Process contrastive batch for validation
        loss = self.contrastive_step(batch)
        self.log('val_contrastive_loss', loss)
        # Also log the val_loss for checkpoint monitoring
        self.log('val_loss', loss)  # This is needed for model checkpointing
        return loss
    
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


# Data module to handle cached datasets
class CachedAudioSimilarityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_specs1,
        train_specs2,
        train_labels,
        val_specs1,
        val_specs2,
        val_labels,
        batch_size=32,
        num_workers=4
    ):
        super().__init__()
        self.train_specs1 = train_specs1
        self.train_specs2 = train_specs2
        self.train_labels = train_labels
        self.val_specs1 = val_specs1
        self.val_specs2 = val_specs2
        self.val_labels = val_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Dataset attributes
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage=None):
        # Create datasets
        self.train_dataset = CachedContrastiveDataset(
            self.train_specs1, self.train_specs2, self.train_labels
        )
        
        self.val_dataset = CachedContrastiveDataset(
            self.val_specs1, self.val_specs2, self.val_labels
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )


# Function to create contrastive CSV files if needed
def create_contrastive_csv_files(
    audio_dir="./Audio_dataset", 
    output_dir="./data",
    train_ratio=0.8,
    similar_ratio=0.5,
    min_pairs_per_speaker=5,
    max_pairs_per_speaker=20
):
    """
    Create contrastive CSV files from an audio directory.
    Expected structure: audio_dir/speaker_name/audio_files.wav
    
    Args:
        audio_dir: Directory containing audio files
        output_dir: Directory to save CSV files
        train_ratio: Ratio of data to use for training
        similar_ratio: Ratio of similar pairs (same speaker)
        min_pairs_per_speaker: Minimum pairs per speaker
        max_pairs_per_speaker: Maximum pairs per speaker
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Output file paths
    train_csv = output_dir / "contrastive_train.csv"
    val_csv = output_dir / "contrastive_val.csv"
    
    # Check if files already exist
    if train_csv.exists() and val_csv.exists():
        print(f"Contrastive CSV files already exist at {train_csv} and {val_csv}")
        return train_csv, val_csv
    
    print("Creating contrastive CSV files...")
    
    # Get all audio files by speaker
    audio_dir = Path(audio_dir)
    speakers = {}
    for speaker_dir in audio_dir.iterdir():
        if speaker_dir.is_dir():
            speaker_name = speaker_dir.name
            audio_files = []
            for audio_file in speaker_dir.glob("*.wav"):
                rel_path = audio_file.relative_to(audio_dir)
                audio_files.append(str(rel_path))
            if audio_files:
                speakers[speaker_name] = audio_files
    
    if not speakers:
        raise ValueError(f"No audio files found in {audio_dir}")
    
    print(f"Found {len(speakers)} speakers")
    
    # Create pairs
    pairs = []
    
    # Similar pairs (same speaker)
    for speaker, files in speakers.items():
        if len(files) < 2:
            continue
        
        num_pairs = min(max_pairs_per_speaker, len(files) * (len(files) - 1) // 2)
        num_pairs = max(min_pairs_per_speaker, num_pairs)
        
        # Create pairs randomly
        file_pairs = []
        for i in range(num_pairs):
            # Sample without replacement if possible
            if len(files) >= 2:
                file1, file2 = random.sample(files, 2)
            else:
                # If only one file, use it twice
                file1 = file2 = files[0]
            
            # Create a similar pair
            file_pairs.append((file1, file2, 1))
        
        pairs.extend(file_pairs)
    
    # Dissimilar pairs (different speakers)
    num_similar = len(pairs)
    num_dissimilar = int(num_similar * (1 - similar_ratio) / similar_ratio)
    
    speaker_list = list(speakers.keys())
    for i in range(num_dissimilar):
        speaker1, speaker2 = random.sample(speaker_list, 2)
        
        # Get random files from each speaker
        if speakers[speaker1] and speakers[speaker2]:
            file1 = random.choice(speakers[speaker1])
            file2 = random.choice(speakers[speaker2])
            
            # Create a dissimilar pair
            pairs.append((file1, file2, 0))
    
    # Shuffle pairs
    random.shuffle(pairs)
    
    # Split into train and validation sets
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    # Save to CSV
    train_df = pd.DataFrame(train_pairs, columns=['audio_path1', 'audio_path2', 'label'])
    val_df = pd.DataFrame(val_pairs, columns=['audio_path1', 'audio_path2', 'label'])
    
    # Save to disk
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    print(f"Created {len(train_df)} training pairs and {len(val_df)} validation pairs")
    print(f"Training CSV saved to {train_csv}")
    print(f"Validation CSV saved to {val_csv}")
    
    return train_csv, val_csv


# Main function for training
def train_similarity_model(
    audio_dir="./Audio_dataset",
    cache_dir="./audio_cache",
    output_dir="./models",
    batch_size=16,
    num_workers=2,
    max_epochs=30,
    lr=0.0001,
    force_recompute=False
):
    # Step 1: Create contrastive CSV files if needed
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True, parents=True)
    
    contrastive_train_csv, contrastive_val_csv = create_contrastive_csv_files(
        audio_dir=audio_dir,
        output_dir=data_dir
    )
    
    # Step 2: Preprocess and cache the dataset
    cache_paths, metadata = preprocess_audio_dataset(
        audio_dir=audio_dir,
        contrastive_train_csv=contrastive_train_csv,
        contrastive_val_csv=contrastive_val_csv,
        cache_dir=cache_dir,
        force_recompute=force_recompute
    )
    
    # Print metadata
    print("Dataset metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Step 3: Load cached data
    train_specs1 = torch.load(cache_paths["train_specs1"])
    train_specs2 = torch.load(cache_paths["train_specs2"])
    train_specs1 = train_specs1.squeeze(1)
    train_specs1 = train_specs1.squeeze(1)
    train_specs1 = train_specs1.transpose(-1, -2)
    
    train_specs2 = train_specs2.squeeze(1)
    train_specs2 = train_specs2.squeeze(1)
    train_specs2 = train_specs2.transpose(-1, -2)
    
    train_labels = torch.load(cache_paths["train_labels"])
    val_specs1 = torch.load(cache_paths["val_specs1"])
    val_specs2 = torch.load(cache_paths["val_specs2"])
    
    # val_specs1 = val_specs1.squeeze(1)
    val_specs1 = val_specs1.squeeze(1)
    val_specs1 = val_specs1.transpose(-1, -2)
    
    # val_specs2 = val_specs2.squeeze(1)
    val_specs2 = val_specs2.squeeze(1)
    val_specs2 = val_specs2.transpose(-1, -2)
    
    val_labels = torch.load(cache_paths["val_labels"])
    
    # Step 4: Create model
    base_model = ResNet(num_classes=52, dropout=0.7)
    model = AudioSimilarityModel(
        base_model=base_model,
        embedding_size=256,
        contrastive_weight=1.0,  # Only using contrastive loss
        triplet_weight=0.0,      # No triplet loss
        contrastive_margin=1.0,
        lr=lr
    )
    
    # Step 5: Create data module
    data_module = CachedAudioSimilarityDataModule(
        train_specs1=train_specs1,
        train_specs2=train_specs2,
        train_labels=train_labels,
        val_specs1=val_specs1,
        val_specs2=val_specs2,
        val_labels=val_labels,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Step 6: Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Step 7: Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='similarity-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_weights_only=True
    )
    
    # Step 8: Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        enable_checkpointing=True
    )
    
    # Step 9: Train model
    print("Starting training...")
    trainer.fit(model, data_module)
    
    # Step 10: Save final model
    final_path = output_dir / "final_audio_similarity_model.ckpt"
    trainer.save_checkpoint(final_path)
    print(f"Training complete! Final model saved to {final_path}")
    
    # Return paths to best model and final model
    return {
        "best_model": checkpoint_callback.best_model_path,
        "final_model": final_path
    }


if __name__ == "__main__":
    # temp = torch.load("./mswc_cache/X.pt")
    # print(temp.shape)
    
    # temp = torch.load("./downstream_cache/train_specs1.pt")
    # print(temp.shape)
    # temp = temp.squeeze(1)
    # temp = temp.squeeze(1)
    # temp = temp.transpose(-1, -2)
    # print(temp.shape)
    # # temp = temp.unsqueeze(1)
    # exit(0)
    # try:
    #     # Start training
    #     model_paths = train_similarity_model(
    #         audio_dir="./Audio_dataset",
    #         cache_dir="./downstream_cache",
    #         output_dir="./downstream_models",
    #         batch_size=16,
    #         num_workers=2,
    #         max_epochs=30,
    #         lr=0.0001,
    #         force_recompute=False
    #     )
        
    #     print(f"Best model path: {model_paths['best_model']}")
    #     print(f"Final model path: {model_paths['final_model']}")
        
    # except Exception as e:
    #     print(f"Error during training: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    
    
    # Testing
    model = AudioSimilarityModel.load_from_checkpoint("./downstream_models/similarity-epoch=10-val_loss=0.1074.ckpt")
    model.eval()
    
    mel_spec = load_and_preprocess_audio_file("./Audios4testing/sam_2.wav", max_duration=1.0)
    mel_spec_tensor = torch.tensor([mel_spec], dtype=torch.float32)
    mel_spec_tensor = mel_spec_tensor.unsqueeze(1)
    mel_spec_tensor = mel_spec_tensor.cuda()
    print(mel_spec_tensor.shape)

    mel_spec1 = load_and_preprocess_audio_file("./Audios4testing/sam_1.wav", max_duration=1.0)

    mel_spec_tensor1 = torch.tensor([mel_spec1], dtype=torch.float32)
    mel_spec_tensor1 = mel_spec_tensor1.unsqueeze(1)
    mel_spec_tensor1 = mel_spec_tensor1.cuda()
    print(mel_spec_tensor1.shape)
         
    
    embs1 = model(mel_spec_tensor)
    embs2 = model(mel_spec_tensor1)
    
    print(f"Embeddings: {embs1.shape}, {embs2.shape}")
    
    print(f"Similarity: {torch.cosine_similarity(embs1, embs2)}")