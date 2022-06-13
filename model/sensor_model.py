from typing import Any
import numpy as np
from torch import nn
import torch


class SensorModel(nn.Module):
    def __init__(self) -> None:
        super(SensorModel, self).__init__()

        # dense network to understand the sensor features
        self.fc = nn.Linear(64, self.num_features)
        self.__model = nn.Sequential(
            nn.Linear(26, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            self.fc
        )

    @property
    def num_features(self) -> int:
        return 26

    def first_layer_params(self) -> np.array:
        """Method extracts the weights of the first layer and returns them.
        """
        # TODO: 
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