import torch.nn as nn
from torch import Tensor

class ModifiedResNet(nn.Module):
    """
    A modified ResNet architecture for Optical Character Recognition (OCR).

    Attributes:
        features (nn.Sequential): A sequential container of the original ResNet layers excluding avgpool and fc layers.
        conv1 (nn.Conv2d): Convolution layer to adjust input channels to 1 (grayscale images).
        post_resnet1 (nn.Conv2d): Convolution layer following the features layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer following post_resnet1.
        relu1 (nn.ReLU): ReLU activation layer following bn1.
        post_resnet2 (nn.Conv2d): Another convolution layer following relu1.
        bn2 (nn.BatchNorm2d): Batch normalization layer following post_resnet2.
        relu2 (nn.ReLU): ReLU activation layer following bn2.
        post_resnet3 (nn.Conv2d): Another convolution layer following relu2.
        bn3 (nn.BatchNorm2d): Batch normalization layer following post_resnet3.
        relu3 (nn.ReLU): ReLU activation layer following bn3.
        dwv (nn.Conv2d): Depthwise convolution layer for channel reduction following relu3.
        lstm1 (nn.LSTM): LSTM layer following the depthwise convolution.
        linear1 (nn.Linear): Fully connected layer to project LSTM output to class scores.
    """

    def __init__(self, original_resnet: nn.Module):
        """
        Initializes the ModifiedResNet with an original_resnet model.

        Args:
            original_resnet (nn.Module): The original ResNet model.
        """
        super(ModifiedResNet, self).__init__()
        self.features = nn.Sequential(*list(original_resnet.children())[:-2]) # Remove avgpool and fc layers in the original resnet
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Adjust input channels to 1, since the input images are grayscale
        
        self.post_resnet1 = nn.Conv2d(512, 512, kernel_size=(2, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(512) # Batch normalization after post_resnet1, important for training
        self.relu1 = nn.ReLU(inplace=True) # ReLU activation after post_resnet1, import for training
        
        self.post_resnet2 = nn.Conv2d(512, 512, kernel_size=(3, 4), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(512) # Batch normalization after post_resnet2, important for training
        self.relu2 = nn.ReLU(inplace=True) # ReLU activation after post_resnet2, import for training
        
        self.post_resnet3 = nn.Conv2d(512, 512, kernel_size=(2, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(512) # Batch normalization after post_resnet3, important for training
        self.relu3 = nn.ReLU(inplace=True) # ReLU activation after post_resnet3, import for training
        
        self.dwv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False) # Depthwise Convolution for channel reduction
        self.lstm1 = nn.LSTM(bidirectional=True, num_layers=2, input_size=128, hidden_size=128, dropout=0)
        self.linear1 = nn.Linear(256, 82) # Project first dimension of LSTM output to 82 (number of classes including the PAD token)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the ModifiedResNet.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = self.features(x)
        x = self.post_resnet1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.post_resnet2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.post_resnet3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.dwv(x)
        
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1).contiguous() # Change the order of the dimensions, this is required for the LSTM layer
        x = x.view(batch_size, height * width, channels) # Reshape to (batch_size, sequence_length, input_dim)
        x, _ = self.lstm1(x)
        x = self.linear1(x)
        return x