from typing import Any, Tuple
import numpy as np
from torch import nn
import torch


class SensorModel(nn.Module):
    def __init__(self) -> None:
        super(SensorModel, self).__init__()

        # dense network to understand the sensor features
        self.__input_layer = nn.Linear(self.num_features, 64)
        self.fc = nn.Linear(256, 128)
        self.__model = nn.Sequential(
            self.__input_layer,
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 196),
            nn.LeakyReLU(),
            nn.Linear(196, 256),
            nn.LeakyReLU(),
            self.fc
        )

    @property
    def num_features(self) -> int:
        return 22

    def first_layer_params(self) -> Tuple[np.array]:
        """Method extracts the weights of the first layer and returns them.
        """
        weights = biases = None
        for name, params in self.__input_layer.named_parameters():
            if name == "weight":
                weights = params.data.cpu().numpy()
            else:
                biases = params.data.cpu().numpy()

        weights_processed = np.sum(np.abs(weights), axis=0)
        biases_included = np.abs(np.dot(weights.T, biases))
        return weights_processed, biases_included
    
    def forward(self, x: torch.tensor) -> Any:
        # reshape the tensor, so the sequence is part of a batch
        batch_size, seq_size, features = x.shape
        x = x.view(-1, features)

        x = self.__model(x)
        
        # reshape the tensor back to have the sequence as a separate
        # dimension
        x = x.view(batch_size, seq_size, 128)

        return x