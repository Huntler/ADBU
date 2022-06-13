from typing import Any
from torch import nn
import torch


class SensorModel(nn.Module):
    def __init__(self) -> None:
        super(SensorModel, self).__init__()

        # dense network to understand the sensor features
        # TODO: remove conv, add dense. Nothing more
        self.__conv_1 = nn.Conv1d(400, 400, 8, 1)
        self.__bn_1 = nn.BatchNorm1d(400)

        self.__conv_2 = nn.Conv1d(400, 400, 4, 1)
        self.__bn_2 = nn.BatchNorm1d(400)

        self.__conv_3 = nn.Conv1d(400, 400, 4, 1) 
        self.__bn_3 = nn.BatchNorm1d(400)

        self.fc = nn.Linear(36, 36)

    @property
    def num_features(self) -> int:
        return 36

    def first_layer_visualization(self, path: str) -> None:
        """Method saves a visualization of the first layer's weights.

        Args:
            path (str): The path where to store the visualization.
        """
        # TODO: use matplotlib
        pass
    
    def forward(self, x) -> Any:
        x = self.__conv_1(x)
        x = self.__bn_1(x)
        x = torch.tanh(x)
        
        x = self.__conv_2(x)
        x = self.__bn_2(x)
        x = torch.tanh(x)

        x = self.__conv_3(x)
        x = self.__bn_3(x)
        x = torch.tanh(x)

        return x