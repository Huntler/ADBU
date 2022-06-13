from typing import Any
from torch import nn
import torch


class SensorModel(nn.Module):
    def __init__(self) -> None:
        super(SensorModel, self).__init__()

        # dense network to understand the sensor features
        self.fc = nn.Linear(64, self.num_features)
        self.__model = nn.Sequential(
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            self.fc
        )

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
    
    def forward(self, x: torch.tensor) -> Any:
        # reshape the tensor, so the sequence is part of a batch
        batch_size, seq_size, features = x.shape
        x = x.view(-1, features)

        x = self.__model(x)
        
        # reshape the tensor back to have the sequence as a separate
        # dimension
        x = x.view(batch_size, seq_size, self.num_features)

        return x