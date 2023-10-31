import os
import pandas as pd
from models.ocrdataset import CustomDataset
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import models,transforms
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
from models.CNNLSTM import ModifiedResNet 
# import albumentations as A
from transformers import PreTrainedTokenizerFast

# Define the tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file = 'C:/Users/ra78lof/occinference/byte-level-BPE.tokenizer.json')

# Add PAD token to the vocabulary, otherwise it will throw an error
tokenizer.add_special_tokens({'pad_token': "pad_token"})
special_tokenizer = tokenizer 

# Define the lightning class, this includes a transfer learning model and a loss function
class OcrModel(pl.LightningModule):
    def __init__(self):
        super().__init__() 
        backbone = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
        encoder_decoder = ModifiedResNet(backbone)
        self.encoder_decoder = encoder_decoder 
        self.loss = nn.CrossEntropyLoss(dim = 2) # CrossEntropyloss on the third dimension  
        self.save_hyperparameters() # Save the hyperparameters for logging purposes

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
        optimizer = optim.SGD(self.parameters(), lr=0.01, weight_decay=0.0001)  
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=100, cycle_momentum=False)
        return [optimizer], [scheduler]
    
class OcrDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_batch_size: int, 
                 val_batch_size: int, 
                 test_batch_size: int):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.tokenizer = special_tokenizer
        self.transforms = transforms.Compose([transforms.Resize((500, 1200)), 
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (0.5,))]) # Normalize the images to the range [-1, 1], this is a grayscale image.

    def prepare_data(self):
        # Load the Data from the excel files, load the images
        self.train_set = CustomDataset(excel_file=os.getcwd() + '/Train_data.xlsx',
                             img_dir=os.getcwd() + '/Train_data/', tokenizer = tokenizer, transform=transforms)
        self.valid_set = CustomDataset(excel_file=os.getcwd() + '/Valid_data.xlsx',
                             img_dir=os.getcwd() + '/Valid_data/', tokenizer = tokenizer, transform=transforms)
        self.test_set = CustomDataset(excel_file=os.getcwd() + '/Test_data.xlsx',
                             img_dir=os.getcwd() + '/Test_data/', tokenizer = tokenizer, transform=transforms)
        

    def set_up(self, stage: str):
        if stage == 'fit':
            self.train_dataset = self.train_set
        if stage == 'validate':
            self.valid_dataset = self.valid_set
        if stage == 'test':
            self.test_dataset = self.test_set
        # Define the train, validation and test datasets we got from prepare_data(), use the transforms we defined above
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.val_batch_size, shuffle=True)

    def test_dataloader(self): 
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=True)

    
dm = OcrDataModule(32, 64, 64)
model = OcrModel()
trainer = pl.Trainer(gpus = 1, max_epochs = 200)
trainer.fit(model, datamodule = dm)
trainer.test(datamodule=dm)
trainer.validate(datamodule=dm)