import argparse
import math
from multiprocessing import freeze_support
import os
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
import torch

# our libraries
from model.base_model import BaseModel
from model.multimodal_model import MultimodalModel
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
    window_size = config_dict["dataset_args"]["window_size"]
    model: BaseModel = config.get_model(model_name)(**config_dict["model_args"], window_size=window_size)
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

    # showing weight analysis before training
    if config_dict["model_name"] == "Multimodal_v1":
        explain_model(model, initial=True)

    # train the model and save it in the end
    model.learn(train, validation, test, epochs=config_dict["train_epochs"],
                save_every=config_dict["save_every"])
    BaseModel.save_to_default(model)

    # explain the model's weights
    if config_dict["model_name"] == "Multimodal_v1":
        explain_model(model)

    # execute the model and look at results
    for X in train:
        sensor, image, label = X
        sensor = sensor.to("cuda")
        image = image.to("cuda")
        label = label.to("cuda")
        pred = model.forward((sensor, image)).argmax(dim=1)
        print(pred, label)


def explain_model(model: MultimodalModel, initial: bool = False):
    path = model.log_path if not initial else f"{model.log_path}/initial"
    if not os.path.exists(path):
        os.mkdir(path)

    sensor_image = model.sensor_image_ratio()
    sensor_importance = model.sensor_importance()

    # pie diagram for sensor-image ratio
    labels = "Sensor", "Image"
    sizes = [100 * sensor_image, 100 * (1-sensor_image)]
    colors = [plt.cm.Reds(.4), plt.cm.Reds(.7)]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    ax1.set_title("Sensor-Image Ratio")
    plt.savefig(f"{path}/sensor_image_ratio.png")

    # bar diagram showing sensor importance
    fig, ax = plt.subplots()
    colors = [plt.cm.Reds(0.36 + .02 * i) for i in range(len(sensor_importance[0]))]
    ax.bar(0.5 + np.arange(len(sensor_importance[0])), sensor_importance[0], color=colors)
    ax.set_xlabel("Sensor ID")
    ax.set_ylabel("Relative importance %")
    ax.set_title("Sensor Importance (no bias)")
    plt.savefig(f"{path}/sensor_importance_weights.png")

    fig, ax = plt.subplots()
    colors = [plt.cm.Reds(0.36 + .02 * i) for i in range(len(sensor_importance[1]))]
    ax.bar(0.5 + np.arange(len(sensor_importance[1])), sensor_importance[1], color=colors)
    ax.set_xlabel("Sensor ID")
    ax.set_ylabel("Relative importance %")
    ax.set_title("Sensor Importance (bias included)")
    plt.savefig(f"{path}/sensor_importance_biases.png")

    # alternative visualisation of sensor importance
    data = np.array(sensor_importance[0])
    data.resize((3, 6), refcheck=False)
    fig, ax = plt.subplots()
    cax = ax.matshow(data, cmap="Reds")
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, f"{j + i * 6}", ha='center', va='center')
    fig.colorbar(cax)
    plt.title("Sensor Importance (no bias)")
    plt.savefig(f"{path}/sensor_importance_weights_alt.png")
    
    data = np.array(sensor_importance[1])
    data.resize((3, 6), refcheck=False)
    fig, ax = plt.subplots()
    cax = ax.matshow(data, cmap="Reds")
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, f"{j + i * 6}", ha='center', va='center')
    fig.colorbar(cax)
    plt.title("Sensor Importance (bias included)")
    plt.savefig(f"{path}/sensor_importance_biases_alt.png")


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
