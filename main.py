import argparse
import math
from multiprocessing import freeze_support
import os

from torch.utils.data import DataLoader
import torch

# our libraries
from model.base_model import BaseModel
from uah_dataset.dataset import Dataset
from utils.config import config


config_dict = None


# initialize dataloader here
def prepare_data(mode: str):
    if mode == "train":
        dataset = Dataset(**config_dict["dataset_args"])
        split_sizes = [int(math.ceil(len(dataset) * 0.8)), int(math.floor(len(dataset) * 0.2))]
        trainset, valset = torch.utils.data.random_split(dataset, split_sizes)

        trainloader = DataLoader(trainset, **config_dict["dataloader_args"])
        validationloader = DataLoader(valset, **config_dict["dataloader_args"])
        return trainloader, validationloader

    if mode == "test":
        dataset = Dataset(**config_dict["dataset_args"])
        dataloader = DataLoader(dataset, pin_memory=True)
        return dataloader


# initialize model here
def prepare_model():
    # load model flag, which decides wether the model is trained or evaluated
    load_flag = False if config_dict["evaluation"] == "None" else True
    log = config_dict["model_args"]["log"]
    config_dict["model_args"]["log"] = False if load_flag else log

    # create the model, by loading its class name
    model_name = config_dict["model_name"]
    model: BaseModel = config.get_model(model_name)(**config_dict["model_args"])
    model.use_device(config_dict["device"])

    # define log path in config and move the current hyperparameters to
    # this directory in the case we have to train the model
    if not load_flag:
        config_dict["evaluation"] = model.log_path
        config.store_args(f"{model.log_path}/config.yml", config_dict)
        print(f"Prepared model: {model_name}")
        return model

    # if we only want to evaluate a model, we have to load the latest saved one
    # from the provided dictionary
    path = config_dict["evaluation"]
    model_versions = []
    for file in os.listdir(path):
        if ".torch" in file:
            model_versions.append(f"{path}/{file}")
    model_versions.sort(reverse=True)

    print(model_versions[0])
    model.load(model_versions[0])

    print(f"Loaded model: {model_name} ({path})")
    return model


def train():
    # prepare the train-, validation- and test datasets / dataloaders
    train, validation = prepare_data(mode="train")
    test = prepare_data(mode="test")

    # prepare the model
    model: BaseModel = prepare_model()

    # train the model and save it in the end
    model.learn(train, validation, test, epochs=config_dict["train_epochs"],
                save_every=config_dict["save_every"])
    BaseModel.save_to_default(model)


def test():
    test = prepare_data(mode="test")
    # TODO


if __name__ == "__main__":
    freeze_support()

    # defining arguments
    parser = argparse.ArgumentParser(description="This program trains and tests a deep " +
                                                 "learning model to detect a driving behaviour.")
    parser.add_argument("--config", dest="train_config", help="Trains a model given the path to a configuration file.")
    args = parser.parse_args()

    # load a configuration file
    config_dict = config.get_args(args.train_config)
    train()
