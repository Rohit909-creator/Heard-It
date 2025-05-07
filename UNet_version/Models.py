import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


# model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#     in_channels=1, out_channels=1, init_features=32, pretrained=False, )

# print(model.named_modules)


class UNet(pl.LightningModule):
    
    def __init__(self, in_channels=1, out_channels=1, init_features=32, pretrained=False):
        super().__init__()
        
        self.Unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=in_channels, out_channels=out_channels, init_features=init_features, pretrained=pretrained)

    def forward(self, X):
        out = self.Unet(X)
        return out
    
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
        loss = self.loss(y_pred, audio)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        
        # Calculate accuracy
        # preds = torch.argmax(y_pred, dim=1)
        # acc = (preds == labels).float().mean()
        # self.log("train_acc", acc, prog_bar=True)
        
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
        loss = self.loss(y_preds, audio)
        
        # Calculate accuracy
        # preds = torch.argmax(y_preds, dim=1)
        # acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", acc, prog_bar=True)
        
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


if __name__ == "__main__":
    
    X = torch.load("./UNet_version/mswc_cache/X.pt")
    # X = torch.randn((2, 1, 32, 32))
    model = UNet(init_features=(32, 40))
    X = X.unsqueeze(1)
    print(X.shape)
    out = model(X[0:2])
    print(out.shape)