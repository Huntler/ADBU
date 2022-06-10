from typing import Any
from torch import nn
import torch
from torchvision.models import resnet50, resnet18


class ImageModel(nn.Module):
    def __init__(self) -> None:
        """This class uses a pre-trained resnet to extract features from images. The 
        output of resnet is passed into a small dense neural network, which is used 
        to focus on relevant features only. The output of this network is the output 
        of the last Linear layer which has 256 nodes. No activation function is applied 
        on the last layer.
        """
        super(ImageModel, self).__init__()

        # load pre-trained resnet and disable training of weigths
        self.__resnet = resnet18(pretrained=True)
        for param in self.__resnet.parameters():
             param.required_grad = False

        # replace the output layer of resnet
        num_features = self.__resnet.fc.in_features
        self.__resnet.fc = nn.Linear(num_features, 256)

        # alternative: conv
        self.__alternative = nn.Sequential(
            nn.Conv2d(3, 12, 3, 1, 0),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(12, 24, 3, 1, 0),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(24, 32, 3, 1, 0),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
    
    def forward(self, x) -> Any:
        batch_size, seq_size, width, height, channels = x.shape
        x = torch.swapaxes(x, -1, -3)

        # for the cnn the batch and sequence counts as one, so combine both
        # view preserves the tensors the original order of the tensor
        x = x.view(-1, channels, height, width)

        x = self.__alternative(x)
        x = torch.relu(x)

        # reshape the data again
        _, features = x.shape
        x = x.view(batch_size, seq_size, features)

        return x