from datetime import datetime
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from utils.tb_logger import CustomMetricsLogger
from tqdm import tqdm


class BaseModel(nn.Module):
    def __init__(self, tag: str, log: bool = True) -> None:
        """This base class defines the training loop to train any neural network. Also, this 
        class handels tensorboard logging and model saving, as well as GPU acceleration if 
        possible.

        Args:
            tag (str): The tag of the model, which is also the logging path.
            log (bool, optional): Enables logging (or disables it if its set to False). 
            Defaults to True.
        """
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
        """This method moves the model to the given device. 

        Args:
            device (str): The device as string ("cpu", "cuda", ...)
        """
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
        torch.save(model.state_dict(), f"{model.log_path}/model_{model_tag}.torch")

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

    def __single_accuracy(self, y_pred, y_test) -> float:
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == torch.flatten(y_test)).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc * 100)

        return acc.detach().cpu().item()

    def learn(self, train, validate=None, test=None, epochs: int = 1, save_every: int = -1, verbose: bool = False):
        """This method trains the model using the trainset. A validation- and testset can be specified 
        to measure the models generalization.

        Args:
            train (_type_): Dataloader of trainset.
            validate (_type_, optional): Dataloader of validationset. Defaults to None.
            test (_type_, optional): Dataloader of testset. Defaults to None.
            epochs (int, optional): Amount of epochs to train. Defaults to 1.
            save_every (int, optional): Save the model each provided epoch, -1 disables this 
            functionality. Defaults to -1.
            verbose (bool, optional): Enables or disables progressbar. Defaults to False.
        """
        # set the model into training mode
        self.train()

        # log the training metrics
        with CustomMetricsLogger(self.writer, parent_tag="Train") as logger:
            # run for n epochs specified
            for e in tqdm(range(epochs)):
                train_iterator = tqdm(train) if verbose else train
                losses = []

                # run for each batch in training set
                for X_sensor, X_image, y in train_iterator:
                    X_sensor = X_sensor.to(self.__device)
                    X_image = X_image.to(self.__device)
                    y = y.to(self.__device).argmax(dim=1)

                    # perform the presiction and measure the loss between the prediction
                    # and the expected output
                    pred_y = self((X_sensor, X_image))

                    # calculate the gradient using backpropagation of the loss
                    loss = self.loss_fn(pred_y, y)
                    
                    # reset the gradient and run backpropagation
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    losses.append(loss.item())
                    logger.sample_log("sample_loss", loss)
                    logger.count(CustomMetricsLogger.SAMPLE, value=X_sensor.size(0))

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

                # save the model every X epoch
                if e % save_every == 0:
                    self.eval()
                    BaseModel.save_to_default(self)
                    self.train()

        self.eval()

    def validate(self, dataloader, step: int) -> float:
        """Method validates model's accuracy based on the given data. In validation, the model
        only looks one step ahead.

        Args:
            dataloader (_type_): The dataloader which contains value, not used for training.
            step (int): The step of the logger.
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
            for X_sensor, X_image, y in dataloader:
                X_sensor = X_sensor.to(self.__device)
                X_image = X_image.to(self.__device)
                y = y.to(self.__device).argmax(dim=1)

                _y = self((X_sensor, X_image))

                y = y.to(self.__device)
                loss = self.loss_fn(_y, y)

                losses.append(loss.item())
                accuracies.append(self.__single_accuracy(_y, y))

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
            for X_sensor, X_image, y in dataloader:
                X_sensor = X_sensor.to(self.__device)
                X_image = X_image.to(self.__device)
                y = y.to(self.__device).argmax(dim=1)

                _y = self((X_sensor, X_image))

                y = y.to(self.__device)
                loss = self.loss_fn(_y, y)

                losses.append(loss.item())
                accuracies.append(self.__single_accuracy(_y, y))

            # calculate some statistics based on the data collected
            accuracy = np.mean(np.array(accuracies))
            loss = np.mean(np.array(losses))

            # log to the tensorboard if wanted
            logger.epoch_log("accuracy", accuracy)
            logger.epoch_log("loss", loss)
        
        self.train()