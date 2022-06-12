# ADBH
Repository of the "Autonomous Driving Behaviour Understanding" project (University Maastricht)

## Setup
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your system and execute the follwing command afterwards.

```$ conda env create -f environment/universal.yml```

After installation, the environment can be activated by calling 

```$ conda activate adbu```

## Prepare dataset

```$ python dataset_example_usage.py --window_size <int: window_size>```

## Train
There are two options to train, the first one trains one model based on a provided configuration file. In order to do so, run

```$ python main.py --config <PATH_TO_CONFIG_FILE>```

To run multiple trainings corresponding to a set of configuration files, type

```$ python utils/mutli_train.py --folder <PATH_TO_CONFIG_FOLDER>```

## Tensorboard
Make sure the conda environment is enabled, then call

```$ tensorboard --logdir=runs```

to show all trainings in tensorboard. Press [here](http://localhost:6006) to access the webpage.
