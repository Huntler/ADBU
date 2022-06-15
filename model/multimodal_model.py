from unicodedata import bidirectional
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
import torch

# our code
from model.base_model import BaseModel
from model.image_model import ImageModel
from model.sensor_model import SensorModel


class MultimodalModel(BaseModel):
    def __init__(self, tag: str, lr: float = 3e-3, lr_decay: float = 9e-1, weight_decay: float = 1e-2, 
                 resnet: bool = True, lstm_layers: int = 2, lstm_hidden: int = 128, log: bool = True) -> None:
        self.writer = None        
        super(MultimodalModel, self).__init__(tag, log)

        # add image model
        self.__image_module = ImageModel(resnet=resnet)

        # add sensor model
        self.__sensor_module = SensorModel()

        # add LSTM
        lstm_in = self.__image_module.num_features + self.__sensor_module.num_features
        self.__lstm = nn.LSTM(lstm_in, lstm_hidden, num_layers=lstm_layers, bidirectional=False, batch_first=True)

        # add classifier output inclunding some dense layers
        self.__dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # define optimizer, loss function and scheduler as BaseModel needs
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=[0.99, 0.999], weight_decay=weight_decay)
        self.scheduler = ExponentialLR(self.optim, gamma=lr_decay)
    
    def sensor_importance(self) -> np.array:
        weights, biases = self.__sensor_module.first_layer_params()
        
        weights_dist = np.exp(weights) / np.sum(np.exp(weights))
        biases_dist = np.exp(biases) / np.sum(np.exp(biases))

        return weights_dist, biases_dist

    def sensor_image_ratio(self) -> float:
        """This method calculates the importance of sensor data vs. image data
        based on then output weights of both modules.

        Returns:
            float: The ration of sensor to image importance.
        """
        # get mean of image fc parameters
        image_mean = 0
        image_fc = self.__image_module.fc
        for name, param in image_fc.named_parameters():
            # name is either 'bias' or 'weight'
            # get params and apply ReLU
            p = param.data.cpu().numpy()
            p[p < 0] = 0

            image_mean += np.mean(p)

        # get mean of sensor fc parameters
        sensor_mean = 0
        sensor_fc = self.__sensor_module.fc
        for name, param in sensor_fc.named_parameters():
            # name is either 'bias' or 'weight'
            # get params and apply ReLU
            p = param.data.cpu().numpy()
            p[p < 0] = 0

            sensor_mean += np.mean(p)
        
        ratio = image_mean / sensor_mean
        return ratio
    
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
        # use: sensor_image_ratio()

        # concatenate both data paths
        x = torch.cat((x_sensor, x_image), -1)

        # pass the combination of both into a LSTM
        x, _ = self.__lstm(x)
        x = x[:, -1, :]
        x = self.__dense(x)

        return x