import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
import torchvision
from tqdm import tqdm
import pandas as pd
from Preprocessing import preprocess_audio_dataset, AudioMelDataset
import numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
class CRNN(pl.LightningModule):
    def __init__(self, mel_bins=40, time_frames=100, num_classes=10, dropout=0.0, learning_rate=0.001):
        """
        Siamese Network for comparing audio embeddings using CNN-based feature extraction
        
        Args:
            embedding_dim (int): Dimension of input embeddings (Whisper embeddings)
            seq_length (int): Sequence length of Whisper embeddings
            output_dim (int): Final embedding dimension
            learning_rate (float): Learning rate for optimizer
        """
        super(CRNN, self).__init__()
        self.lr = learning_rate
        self.num_classes = num_classes
        self.mel_bins = mel_bins
        
        
        
        self.conv1 = nn.Conv2d(1, 16, (3,3))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d((2,2))
        
        self.conv2 = nn.Conv2d(16, 32, (3,3))
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2,2))
        
        self.conv3 = nn.Conv2d(32, 64, (3,3))
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool3 = nn.MaxPool2d((1,2))
        
        self.dropout = nn.Dropout(0.4)
        # self.gru1 = nn.GRU(64*(mel_bins//8), 64, batch_first=True)
        # self.gru2 = nn.GRU(64, 64, batch_first=True)
        self.fc = nn.Linear(64*(self.mel_bins)*3, (64*(self.mel_bins)*3)//2)
        self.linear_map = nn.Linear((64*(self.mel_bins)*3)//2, num_classes)
        
        # Loss function for training
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x):
        """
        Forward pass for one input
        
        Args:
            x: Input embedding of shape [batch_size, seq_length, 1, embedding_dim]
        """
        B, _, _, _ = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        # x = x.view(-1, 64*(self.mel_bins//8))
        # print(x.shape)
        x = x.view(B, 64*(self.mel_bins)*3)
        # print(x.shape)
        # x = self.gru1(x)
        # x = self.gru2(x)
        x = self.dropout(self.fc(x))
        x = self.linear_map(x)
        
        return x

    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Tuple of (inputs, labels)
                  inputs: Tuple of (audio1, audio2)
                  labels: 1 for similar pair, 0 for dissimilar pair
        """
        # print(batch)
        audio, labels = batch
        
        
        # Forward pass
        y_pred = self(audio)
        
        # Calculate loss
        loss = self.loss(y_pred, labels)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        
        Args:
            batch: Tuple of (inputs, labels)
                  inputs: Tuple of (audio1, audio2)
                  labels: 1 for similar pair, 0 for dissimilar pair
        """
        audio, labels = batch
        
        # Forward pass
        y_preds = self(audio)
        
        # Calculate loss
        loss = self.loss(y_preds, labels)
        
        # # Calculate distance
        # distance = F.pairwise_distance(embeddings1, embeddings2)
        
        # # Calculate accuracy (simple threshold-based)
        # predictions = (distance < 0.5).float()
        # accuracy = (predictions == labels).float().mean()
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", accuracy, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.5, patience=5, verbose=True
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            # "monitor": "val_loss"
        }

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

# def load_data(filepath):
#     df = pd.read_csv(filepath)
#     # Assuming your CSV has columns 'text' and 'label'
#     # If different, adjust the column names below
#     texts = df['Text'].values
#     labels = df['Label'].values
#     return texts, labels

# # Preprocess the text
# def data_prep(filepath, validation_split=0.2):    
#     pass


# Create dummy dataset
def create_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # X, y, label_encoder, max_length, max_duration = preprocess_audio_dataset(
    #     audio_dir="Audio_dataset2", 
    #     cache_dir="./mswc_cache"
    # )
    # X = np.expand_dims(X, 1)
    X = torch.load("./mswc_cache/X.pt", weights_only=True)
    X = X.unsqueeze(1)
    y = torch.load("./mswc_cache/y.pt", weights_only=True)
    print("Shapes: ",X.shape, y.shape)
    # Create dataset and dataloader
    # dataset = AudioMelDataset(X, y)
    # return dataset
    return TensorDataset(X, y)


# Load data into DataLoaders
def get_dataloaders(batch_size=32):
    dataset = create_data()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # print(train_size, val_size)
    # Create a generator that matches your device
    generator = torch.Generator()
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# Initialize and train model
def train_model():
    # input_size, hidden_size, output_size = 256, 512, 2
    # Maximum audio duration: 4.14 seconds
    with open("./mswc_cache/classes.txt", 'r') as f:
        s = f.read()
    num_classes = len(s.split("\n"))
    print(f"Num Classes: {num_classes}")
    # model = CRNN(num_classes=num_classes)
    model = ResNetMel(num_classes=num_classes, dropout=0.7)
    print(model.named_modules)
    train_loader, val_loader = get_dataloaders()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, filename="2epochmodel"
    )

    # trainer = pl.Trainer(max_epochs=2, accelerator="gpu" if torch.cuda.is_available() else "cpu",
    #                      callbacks=[checkpoint_callback],
    #                     enable_progress_bar=True,  # Disable default tqdm ba
    #                     )
    
    trainer = pl.Trainer(max_epochs=15,
                        enable_progress_bar=True,  # Disable default tqdm ba
                        num_nodes=1,
                        enable_checkpointing=True
                        )
    
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    model = train_model()
    # model = CRNN()
    # model = ResNetMel()
    # print(model.named_modules)
    # x = torch.randn((2, 1, 176, 40))
    # out = model(x)
    # print(out)
    # inp = torch.randn((2, 1500, 1, 384))
    # out = model(inp, inp)
    # print(out[0].shape)
    # data_prep("dataversion2.csv")
    # get_dataloaders()