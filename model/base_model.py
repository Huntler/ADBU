from datetime import datetime
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from model.tb_logger import CustomMetricsLogger
from tqdm import tqdm


class BaseModel(nn.Module):
    def __init__(self, tag: str, log: bool = True) -> None:
        super(BaseModel, self).__init__()

        # enable tensorboard
        if self.writer is None:
            self.__tb_sub = datetime.now().strftime("%m-%d-%Y_%H%M%S")
            self.__tb_path = f"runs/{tag}/{self.__tb_sub}"
            self.writer = SummaryWriter(self.__tb_path)

        # check for gpu, only inform the user
        self.__device = "cpu"
        self.__device_name = "CPU"
        if torch.cuda.is_available():
            self.__device_name = torch.cuda.get_device_name(0)
            print(f"GPU acceleration available on {self.__device_name}")

        # define object which where defined by children of this class
        self.scheduler: _LRScheduler = None
        self.optim: Optimizer = None
        self.loss_fn: Module = None

    @property
    def log_path(self) -> str:
        return self.__tb_path
    
    @property
    def device(self) -> str:
        return self.__device
    
    @property
    def device_name(self) -> str:
        return self.__device_name

    def use_device(self, device: str) -> None:
        self.__device = device
        self.to(self.__device)

    @staticmethod
    def save_to_default(model) -> None:
        """This method saves the current model state to the tensorboard 
        directory.
        """
        # move the model to CPU first
        dev = model.device
        model.use_device("cpu")

        # save the model
        model_tag = datetime.now().strftime("%H%M%S")
        torch.save(model.state_dict(), f"{model._tb_path}/model_{model_tag}.torch")

        # and move it back to the original device
        model.use_device(dev)

    def load(self, path) -> None:
        raise NotImplementedError()

    def forward(self, X):
        """
        This method performs the forward call on the neural network 
        architecture.

        Args:
            X (Any): The input passed to the defined neural network.

        Raises:
            NotImplementedError: The Base model has not implementation 
                                 for this.
        """
        raise NotImplementedError

    def learn(self, train, validate=None, test=None, epochs: int = 1, verbose: bool = False):
        # set the model into training mode
        self.train()

        # log the training metrics
        with CustomMetricsLogger(self.writer, parent_tag="Train") as logger:
            # run for n epochs specified
            for e in tqdm(range(epochs)):
                train_iterator = tqdm(train) if verbose else train
                losses = []

                # run for each batch in training set
                for X, y in train_iterator:
                    X = X.to(self.__device)
                    y = y.to(self.__device)

                    # perform the presiction and measure the loss between the prediction
                    # and the expected output
                    pred_y = self(X)

                    # calculate the gradient using backpropagation of the loss
                    loss = self.loss_fn(pred_y, y)
                    
                    # reset the gradient and run backpropagation
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    losses.append(loss.item())
                    logger.sample_log("sample_loss", loss)
                    logger.count(CustomMetricsLogger.SAMPLE, value=X.size(0))

                # log for each batch we trained
                mean_loss = np.mean(losses, axis=0)
                logger.epoch_log("epoch_loss", mean_loss)

                # if there is an adaptive learning rate (scheduler) available
                if self.scheduler:
                    self.scheduler.step()
                    lr = self.scheduler.get_last_lr()[0]
                    logger.epoch_log("learning_rate", lr)

                # run a validation of the current model state
                if validate:
                    self.validate(validate, logger.epoch)
                
                # run a test of the current model state
                if test:
                     self.test(test, logger.epoch)

                logger.count(CustomMetricsLogger.EPOCH)

        self.eval()

    def validate(self, dataloader, step: int) -> float:
        """Method validates model's accuracy based on the given data. In validation, the model
        only looks one step ahead.

        Args:
            dataloader (_type_): The dataloader which contains value, not used for training.
            step (int): The step of the logger.

        Returns:
            float: The model's accuracy.
        """
        self.eval()
        accuracies = []
        losses = []

        # log to the tensorboard
        with CustomMetricsLogger(self.writer, "Validation") as logger:
            # set the logger's pointer position
            logger.count(CustomMetricsLogger.EPOCH, step)

            # predict all y's of the validation set and append the model's accuracy 
            # to the list
            for X, y in dataloader:
                _y = self.predict(X, as_list=False)

                y = y.to(self.__device)
                loss = self.loss_fn(_y, y)

                losses.append(loss.item())
                accuracies.append(1 - loss.item())

            # calculate some statistics based on the data collected and log them
            accuracy = np.mean(np.array(accuracies))
            loss = np.mean(np.array(losses))

            logger.epoch_log("accuracy", accuracy)
            logger.epoch_log("loss", loss)

        self.train()

    def test(self, dataloader, step: int) -> float:
        """Method validates model's accuracy based on the given data. In validation, the model
        only looks one step ahead.

        Args:
            dataloader (_type_): The dataloader which contains value, not used for training.
            step (int): The step of the logger.

        Returns:
            float: The model's accuracy.
        """
        self.eval()
        accuracies = []
        losses = []

        # log to the tensorboard
        with CustomMetricsLogger(self.writer, "Test") as logger:
            # set the logger's pointer position
            logger.count(CustomMetricsLogger.EPOCH, step)
            
            # predict all y's of the validation set and append the model's accuracy 
            # to the list
            for X, y in dataloader:
                _y = self.predict(X, as_list=False)

                y = y.to(self.__device)
                loss = self.loss_fn(_y, y)

                losses.append(loss.item())
                accuracies.append(1 - loss.item())

            # calculate some statistics based on the data collected
            accuracy = np.mean(np.array(accuracies))
            loss = np.mean(np.array(losses))

            # log to the tensorboard if wanted
            logger.epoch_log("accuracy", accuracy)
            logger.epoch_log("loss", loss)
        
        self.train()