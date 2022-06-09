from typing import Any
from torch import nn
import torch
from torchvision.models import resnet50


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
        self.__resnet = resnet50(pretrained=True)
        for param in self.__resnet.parameters():
            param.required_grad = False

        # add dense layers to the output of resnet, those were tuned
        # to focus on needed features extracted by resnet
        self.__dense_1 = nn.Linear(512, 384)
        self.__dense_2 = nn.Linear(384, 256)
    
    def forward(self, x) -> Any:
        x = self.__resnet(x)
        x = torch.relu(x)

        x = self.__dense_1(x)
        x = torch.relu(x)

        x = self.__dense_2(x)
        return x