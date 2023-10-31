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
import albumentations as A


# Define the lightning class, this includes a transfer learning model and a loss function
class OcrModel(pl.LightningModule):
    def __init__(self):
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
    
class OcrDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_batch_size: int, 
                 val_batch_size: int, 
                 test_batch_size: int,
                 tokenizer = None, 
                 feature_extractor = None, 
                 transforms=None, 
                 max_target_length = 45):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.transforms = transforms.Compose([transforms.Resize((500, 1200)), 
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (0.5,))]) # Normalize the images to the range [-1, 1], this is a grayscale image.

    def prepare_data(self):
        # Load the Data from the excel files, load the images
        train_set = pd.read_excel(os.getcwd() + '/data/dom_project/Train_data.xlsx')
        valid_set = pd.read_excel(os.getcwd() + '/data/dom_project/Val_data.xlsx')
        test_set = pd.read_excel(os.getcwd() + '/data/dom_project/Test_data.xlsx')  
        img_names = train_set['Image_name'].tolist()

    def set_up(self):
        # Define the train, validation and test datasets we got from prepare_data(), use the transforms we defined above
        pass

    def train_dataloader(self):
        train_dataset = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        valid_dataset = DataLoader(valid_dataset, batch_size=self.val_batch_size, shuffle=True)

    def test_dataloader(self): 
        test_dataset = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=True)




# Define the dataloader
train_dataset = CustomDataset(excel_file='C:/Users/ra78lof/occinference/Test_data.xlsx',
                             img_dir='C:/Users/ra78lof/occinference/Test_data/', tokenizer = tokenizer, transform=transforms)

valid_dataset = CustomDataset(excel_file='C:/Users/LMMISTA-WAP265/OcciGen/data/dom_project/Val_data.xlsx',
                              img_dir='C:/Users/LMMISTA-WAP265/OcciGen/data/dom_project/Val_data/', tokenizer = tokenizer, transform=transforms)

test_dataset = CustomDataset(excel_file='C:/Users/ra78lof/occinference/Test_data.xlsx',
                             img_dir='C:/Users/ra78lof/occinference/Test_data/', tokenizer = tokenizer, transform=transforms)
    

model = OcrModel()
trainer = pl.Trainer(gpus = 1, max_epochs = 200)
trainer.fit(model, train_dataset, valid_dataset)