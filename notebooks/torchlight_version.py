import os
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from torch import optim, nn, utils, Tensor
from torchvision import models
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
from models.CNNLSTM import ModifiedResNet 


# Define the lightning class, this includes a transfer learning model and a loss function
class OcrModel(pl.LightningModule):
    def __init__(self, original_resnet, encoder_decoder):
        super().__init__() 
        backbone = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
        encoder_decoder = ModifiedResNet(backbone)
        self.encoder_decoder = encoder_decoder 
        self.loss = nn.CrossEntropyLoss(dim = 2) # CrossEntropyloss on the third dimension  

    def training_step(self, batch, batch_idx):  
        x, y = batch
        y_hat = self.encoder_decoder(x)
        loss = self.loss(y_hat, y)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.encoder_decoder(x)
        loss = self.loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=0.0001)  
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=100, cycle_momentum=False)
        return [optimizer], [scheduler]
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()
    

    

