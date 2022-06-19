from model.base_model import BaseModel
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.models import resnet50, resnet18
import torch


class FullMultimodalModel(BaseModel):
    def __init__(self, tag: str, lr: float = 3e-3, lr_decay: float = 9e-1, weight_decay: float = 1e-2, momentum: float = 0.9,
                 resnet: bool = True, lstm_layers: int = 2, lstm_hidden: int = 128, log: bool = True) -> None:
        self.writer = None        
        super().__init__(tag, log)

        # image model
        if resnet:
            # load pre-trained resnet and disable training of weigths
            __resnet = resnet18(pretrained=True)
            for param in __resnet.parameters():
                param.required_grad = False

            # replace the output layer of resnet
            num_features = __resnet.fc.in_features
            __resnet.fc = nn.Linear(num_features, 256)
            self.__image_model = __resnet
            self.__image_fc = __resnet.fc

        else:
            # alternative: conv
            __alt_fc = nn.Linear(512, self.num_features)
            __alternative = nn.Sequential(
                nn.Conv2d(3, 12, 7, 3, 0),
                nn.BatchNorm2d(12),
                nn.MaxPool2d(2),
                nn.Tanh(),
                nn.Conv2d(12, 24, 5, 2, 0),
                nn.BatchNorm2d(24),
                nn.MaxPool2d(2),
                nn.Tanh(),
                nn.Conv2d(24, 32, 3, 1, 0),
                nn.BatchNorm2d(32),
                nn.Flatten(),
                nn.Linear(1152, 768),
                nn.ReLU(),
                nn.Linear(768, 512),
                nn.ReLU(),
                __alt_fc
            )

            self.__image_model = __alternative      
            self.__image_fc = __alt_fc

        # sensor model        
        self.__sensor_fc = nn.Linear(256, 128)
        self.__sensor_model = nn.Sequential(
            nn.Linear(22, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 196),
            nn.ReLU(),
            nn.Linear(196, 256),
            nn.ReLU(),
            self.__sensor_fc
        )

        # add LSTM
        lstm_in = 256 + 128
        self.__lstm = nn.LSTM(lstm_in, lstm_hidden, num_layers=lstm_layers, bidirectional=False, batch_first=True)
        self.__lstm_layers = lstm_layers
        self.__hidden_dim = lstm_hidden

        # add classifier output inclunding some dense layers
        self.__dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # define optimizer, loss function and scheduler as BaseModel needs
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # self.optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=[0.99, 0.999], weight_decay=weight_decay)
        self.optim = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = ExponentialLR(self.optim, gamma=lr_decay)

    def forward(self, X):
        x_sensor, x_image = X

        # pass the image data through the image model
        batch_size, seq_size, width, height, channels = x_image.shape
        x = torch.swapaxes(x_image, -1, -3)

        # for the cnn the batch and sequence counts as one, so combine both
        # view preserves the tensors the original order of the tensor
        x = x.view(-1, channels, height, width)

        x = self.__image_model(x)
        x = torch.relu(x)

        # reshape the data again
        _, features = x.shape
        x = x.view(batch_size, seq_size, features)
        x_image = torch.relu(x)

        # pass the sensor data through the sensor model
        # reshape the tensor, so the sequence is part of a batch
        batch_size, seq_size, features = x_sensor.shape
        x = x_sensor.view(-1, features)

        x = self.__sensor_model(x)
        
        # reshape the tensor back to have the sequence as a separate
        # dimension
        x = x.view(batch_size, seq_size, 128)
        x_sensor = torch.relu(x)

        # the mean of the last weights of x_sensor and x_image can be compared
        # to determine the image- or sensordata importance
        # use: sensor_image_ratio()

        # concatenate both data paths
        x = torch.cat((x_sensor, x_image), -1)

        # pass the combination of both into a LSTM
        # hidden = self.__init_hidden(x.shape[0])
        # x, hidden = self.__lstm(x, hidden)
        x, _ = self.__lstm(x)
        x = x[:, -1, :]
        x = self.__dense(x)

        return x