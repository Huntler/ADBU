import torch
import numpy as np
import os
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        #provide with window size of data you want to load
        window_size = 4


        # load all matrices
        self.sensor_data = self.read_sensor(window_size)
        self.image_data = 0 #np.random.rand(2937, 4, 2, 2, 1) # TODO PHILLIP load the correct dataset(based on window_size)
        self.labels = self.read_labels(window_size)
    def __len__(self) -> int:
        # amount of total samples / windows / whatever we train
        if (len(self.sensor_data) != len(self.image_data)):
            raise RuntimeError("Sensor data and video data have different sizes")
            return -1
        return len(self.labels)

    def __getitem__(self, index):
        # TODO maybe do a check before returning
        #print(self.labels)
        # image = np.load('path', ) # (1, window_size, 32,32,3) #TODO PHILLIP
        return (self.sensor_data[index], 0 , self.labels[index]) # TODO note it might crush hereself.image_data[index])

    def read_sensor(self, window_size):
        dat_dir = '..\\uah_dataset\\processed_dataset\\sensor\\dat\\window_' + str(window_size)
        # get shape
        files = os.listdir(dat_dir)
        # read data
        file = open(dat_dir + "\\" + files[1], 'rb')
        shapes = pickle.load(file)
        return np.memmap(dat_dir + "\\" + files[2], dtype='float32', mode='r', shape=shapes['sensor'])
    def read_labels(self, window_size):
        dat_dir = '..\\uah_dataset\\processed_dataset\\sensor\\dat\\window_' + str(window_size)
        # get shape
        files = os.listdir(dat_dir)
        # read data
        file = open(dat_dir + "\\" + files[1], 'rb')
        shapes = pickle.load(file)
        return np.memmap(dat_dir + "\\" + files[0], dtype='int', mode='r', shape=shapes['labels'])
if __name__ == "__main__":
    # TODO: perform our tests
    d = Dataset()

    sensor, image, label = d[0:2]
    print(sensor)
    #print(image)
    print(label)

