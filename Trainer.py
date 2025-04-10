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

class SiameseNetwork(pl.LightningModule):
    def __init__(self, embedding_dim=384, seq_length=1500, output_dim=768, learning_rate=0.001):
        """
        Siamese Network for comparing audio embeddings using CNN-based feature extraction
        
        Args:
            embedding_dim (int): Dimension of input embeddings (Whisper embeddings)
            seq_length (int): Sequence length of Whisper embeddings
            output_dim (int): Final embedding dimension
            learning_rate (float): Learning rate for optimizer
        """
        super(SiameseNetwork, self).__init__()
        self.lr = learning_rate
        self.output_dim = output_dim
        
        # CNN-based feature extraction network
        # Input shape: [batch_size, seq_length, 1, embedding_dim]
        self.feature_extractor = nn.Sequential(
            # Reduce sequence dimension in stages
            nn.Conv2d(seq_length, seq_length // 3, kernel_size=1),
            nn.BatchNorm2d(seq_length // 3),
            nn.ReLU(),
            
            nn.Conv2d(seq_length // 3, seq_length // 10, kernel_size=1),
            nn.BatchNorm2d(seq_length // 10),
            nn.ReLU(),
            
            nn.Conv2d(seq_length // 10, 10, kernel_size=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )
        
        # Flattened dimension: 10 * 1 * embedding_dim
        flat_dim = 10 * 1 * embedding_dim
        
        # Final fully connected layers to get the desired output dimension
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # Loss function for training
        self.contrastive_loss = ContrastiveLoss(margin=1.0)
        
    def forward_one(self, x):
        """
        Forward pass for one input
        
        Args:
            x: Input embedding of shape [batch_size, seq_length, 1, embedding_dim]
        """
        x = self.feature_extractor(x)
        x = self.fc(x)
        # L2 normalize the output embeddings
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def forward(self, input1, input2):
        """
        Forward pass for Siamese network
        
        Args:
            input1: First audio embedding [batch_size, seq_length, 1, embedding_dim]
            input2: Second audio embedding [batch_size, seq_length, 1, embedding_dim]
        
        Returns:
            output1: Transformed embedding for first input
            output2: Transformed embedding for second input
        """
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Tuple of (inputs, labels)
                  inputs: Tuple of (audio1, audio2)
                  labels: 1 for similar pair, 0 for dissimilar pair
        """
        # print(batch)
        audio1, audio2, labels = batch
        
        
        # Forward pass
        embeddings1, embeddings2 = self(audio1, audio2)
        
        # Calculate loss
        loss = self.contrastive_loss(embeddings1, embeddings2, labels)
        
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
        audio1, audio2, labels = batch
        
        # Forward pass
        embeddings1, embeddings2 = self(audio1, audio2)
        
        # Calculate loss
        loss = self.contrastive_loss(embeddings1, embeddings2, labels)
        
        # Calculate distance
        distance = F.pairwise_distance(embeddings1, embeddings2)
        
        # Calculate accuracy (simple threshold-based)
        predictions = (distance < 0.5).float()
        accuracy = (predictions == labels).float().mean()
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        
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


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for Siamese networks
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Args:
            output1: First embedding
            output2: Second embedding
            label: 1 if embeddings should be similar, 0 if they should be dissimilar
        """
        # Calculate euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Contrastive loss
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive

        

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
    X1 = torch.randn((2, 1500, 1, 384))
    X2 = torch.randn((2, 1500, 1, 384))
    Y = torch.zeros((2,))
    Y = Y.to(torch.float32)
    # print(X[0])
    # print(X.shape, Y.shape)
    return TensorDataset(X1, X2, Y)


# Load data into DataLoaders
def get_dataloaders(batch_size=32):
    dataset = create_data()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # print(train_size, val_size)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# Initialize and train model
def train_model():
    # input_size, hidden_size, output_size = 256, 512, 2
    model = SiameseNetwork()

    train_loader, val_loader = get_dataloaders()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, filename="2epochmodel"
    )

    trainer = pl.Trainer(max_epochs=2, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         callbacks=[checkpoint_callback],
                        enable_progress_bar=True  # Disable default tqdm ba
                        )
    
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    model = train_model()
    # model = SiameseNetwork()
    # inp = torch.randn((2, 1500, 1, 384))
    # out = model(inp, inp)
    # print(out[0].shape)
    # data_prep("dataversion2.csv")
    # get_dataloaders()