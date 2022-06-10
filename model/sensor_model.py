from typing import Any
from torch import nn
import torch


class SensorModel(nn.Module):
    def __init__(self) -> None:
        super(SensorModel, self).__init__()

        # 1D convolutional to detect patterns within one feature
        self.__conv_1 = nn.Conv1d(400, 400, 10, stride=2)
        self.__bn_1 = nn.BatchNorm1d(200)

        self.__conv_2 = nn.Conv1d(200, 200, 10, stride=2)
        self.__bn_1 = nn.BatchNorm1d(100)

        self.__conv_3 = nn.Conv1d(100, 100, 10, stride=2)        
        self.__bn_1 = nn.BatchNorm1d(100)
    
    def forward(self, x) -> Any:
        x = self.__conv_1(x)
        x = nn.MaxPool1d(2)
        x = self.__bn_1(x)
        x = torch.tanh(x)
        
        x = self.__conv_2(x)
        x = nn.MaxPool1d(2)
        x = self.__bn_2(x)
        x = torch.tanh(x)

        x = self.__conv_3(x)
        x = self.__bn_2(x)
        x = torch.tanh(x)

        return x