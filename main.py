import argparse
from multiprocessing import freeze_support
from utils.config import config


config_dict = None


# TODO: initialize dataloader here
# TODO: initialize model here
# TODO: add train methods here


if __name__ == "__main__":
    freeze_support()

    # defining arguments
    parser = argparse.ArgumentParser(description="This program trains and tests a deep " +
                                                 "learning model to detect a driving behaviour.")
    parser.add_argument("--config", dest="train_config", help="Trains a model given the path to a configuration file.")
    args = parser.parse_args()

    # load a configuration file
    config_dict = config.get_args(args.train_config)
