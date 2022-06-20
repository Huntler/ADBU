import numpy as np
import torch
from model.base_model import BaseModel
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR


class SensorOnly(BaseModel):
    def __init__(self, tag: str, lr: float = 3e-3, lr_decay: float = 9e-1, weight_decay: float = 1e-2, momentum: float = 0.9,
                 resnet: bool = True, lstm_layers: int = 2, lstm_hidden: int = 128, dropout: float = 0.0, window_size: int = 30, 
                 log: bool = True) -> None:
        self.writer = None        
        super().__init__(tag, log)

        # sensor model        
        self.__sensor_fc = nn.Linear(64, 128)
        self.__sensor_model = nn.Sequential(
            nn.Linear(22, 64),
            nn.LeakyReLU(),
            self.__sensor_fc
        )

         # add LSTM
        self.__lstm = nn.LSTM(128, lstm_hidden, num_layers=lstm_layers, bidirectional=False, dropout=dropout, batch_first=True)

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
        self.optim = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = ExponentialLR(self.optim, gamma=lr_decay)

    def sensor_importance(self) -> np.array:
        # get the weights of the first layer
        weights = biases = None
        for name, params in self.__sensor_model.named_parameters():
            if name == "weight":
                weights = params.data.cpu().numpy()
            else:
                biases = params.data.cpu().numpy()
            
            if weights != None and biases != None:
                break

        weights_processed = np.sum(np.abs(weights), axis=0)
        biases_included = np.abs(np.dot(weights.T, biases))
        
        # calculate distribution of weights
        weights_dist = np.exp(weights_processed) / np.sum(np.exp(weights_processed))
        biases_dist = np.exp(biases_included) / np.sum(np.exp(biases_included))
        return weights_dist, biases_dist
    
    def sensor_image_ratio(self) -> float:
        return 1

    def forward(self, X):
        x, _ = X

        # pass the sensor data through the sensor model
        # reshape the tensor, so the sequence is part of a batch
        batch_size, seq_size, features = x.shape
        x = x.view(-1, features)

        x = self.__sensor_model(x)
        
        # reshape the tensor back to have the sequence as a separate
        # dimension
        x = x.view(batch_size, seq_size, 128)
        x = torch.relu(x)

        # pass the combination of both into a LSTM
        # hidden = self.__init_hidden(x.shape[0])
        # x, hidden = self.__lstm(x, hidden)
        x, _ = self.__lstm(x)
        # x = x[:, -1, :]
        x = self.__dense(x)

        return x