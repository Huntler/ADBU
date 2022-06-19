from typing import Tuple
from unicodedata import bidirectional
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from torch import dropout, nn
import torch
import torch.autograd as autograd

# our code
from model.base_model import BaseModel
from model.image_model import ImageModel
from model.sensor_model import SensorModel


class MultimodalModel(BaseModel):
    def __init__(self, tag: str, lr: float = 3e-3, lr_decay: float = 9e-1, weight_decay: float = 1e-2, momentum: float = 0.9,
                 resnet: bool = True, lstm_layers: int = 2, lstm_hidden: int = 128, dropout: float = 0.0, window_size: int = 30, 
                 log: bool = True) -> None:
        self.writer = None        
        super(MultimodalModel, self).__init__(tag, log)

        self.__submodels = nn.ModuleDict({
            "image": ImageModel(resnet=resnet),
            "sensor": SensorModel()
        })

        # add LSTM
        lstm_in = self.__submodels["image"].num_features + 128
        self.__lstm = nn.LSTM(lstm_in, lstm_hidden, num_layers=lstm_layers, bidirectional=False, dropout=dropout, batch_first=True)
        self.__lstm_layers = lstm_layers
        self.__hidden_dim = lstm_hidden

        # add classifier output inclunding some dense layers
        conv_out_size = (lstm_hidden - 8) * (window_size // 8)
        self.__dense = nn.Sequential(
            nn.Conv1d(window_size, window_size // 4, 5, 1, 0),
            nn.BatchNorm1d(window_size // 4),
            nn.Tanh(),
            nn.Conv1d(window_size // 4, window_size // 8, 5, 1, 0),
            nn.BatchNorm1d(window_size // 8),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 3)
        )

        # define optimizer, loss function and scheduler as BaseModel needs
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # self.optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=[0.99, 0.999], weight_decay=weight_decay)
        # params = list(self.parameters()) + list(self.__image_module.parameters()) + list(self.__sensor_module.parameters())
        self.optim = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = ExponentialLR(self.optim, gamma=lr_decay)
    
    def __init_hidden(self, batch_size) -> Tuple[torch.tensor]:
        return (autograd.Variable(torch.zeros(self.__lstm_layers, batch_size, self.__hidden_dim, device=self.device)),
                autograd.Variable(torch.zeros(self.__lstm_layers, batch_size, self.__hidden_dim, device=self.device)))

    def sensor_importance(self) -> np.array:
        weights, biases = self.__submodels["sensor"].first_layer_params()
        
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
        image_fc = self.__submodels["image"].fc
        for name, param in image_fc.named_parameters():
            # name is either 'bias' or 'weight'
            # get params and apply ReLU
            p = param.data.cpu().numpy()
            p[p < 0] = 0

            image_mean += np.mean(p)

        # get mean of sensor fc parameters
        sensor_mean = 0
        sensor_fc = self.__submodels["sensor"].fc
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
        x_image = self.__submodels["image"](x_image)
        x_image = torch.relu(x_image)

        # pass the sensor data through the sensor model
        x_sensor = self.__submodels["sensor"](x_sensor)
        x_sensor = torch.relu(x_sensor)

        # the mean of the last weights of x_sensor and x_image can be compared
        # to determine the image- or sensordata importance
        # use: sensor_image_ratio()

        # concatenate both data paths
        x = torch.cat((x_sensor, x_image), -1)

        # pass the combination of both into a LSTM
        # hidden = self.__init_hidden(x.shape[0])
        # x, hidden = self.__lstm(x, hidden)
        x, _ = self.__lstm(x)
        x = self.__dense(x)

        return x