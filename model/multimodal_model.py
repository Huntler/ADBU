from unicodedata import bidirectional
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
import torch

# our code
from model.base_model import BaseModel
from model.image_model import ImageModel
from model.sensor_model import SensorModel


class MultimodalModel(BaseModel):
    def __init__(self, tag: str, log: bool = True) -> None:
        self.writer = None        
        super(MultimodalModel, self).__init__(tag, log)

        # add image model
        self.__image_module = ImageModel()

        # add sensor model
        self.__sensor_module = SensorModel()

        # add LSTM
        self.__lstm = nn.LSTM(279, 128, num_layers=2, dropout=0.1, bidirectional=False, batch_first=True)

        # add classifier output inclunding some dense layers
        self.__dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400 * 128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # define optimizer, loss function and scheduler as BaseModel needs
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.AdamW(self.parameters(), lr=0.003, betas=[0.99, 0.999], weight_decay=0.05)
        self.scheduler = ExponentialLR(self.optim, gamma=0.9)
    
    def forward(self, X):
        x_sensor, x_image = X

        # pass the image data through the image model
        x_image = self.__image_module(x_image)
        x_image = torch.relu(x_image)

        # pass the sensor data through the sensor model
        x_sensor = self.__sensor_module(x_sensor)
        x_sensor = torch.relu(x_sensor)

        # the mean of the last weights of x_sensor and x_image can be compared
        # to determine the image- or sensordata importance

        # concatenate both data paths
        x = torch.cat((x_sensor, x_image), -1)

        # pass the combination of both into a LSTM
        x, _ = self.__lstm(x)
        x = self.__dense(x)

        return x