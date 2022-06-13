import torch
import numpy as np
import os
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        #provide with window size of data you want to load
        self.window_size = 10


        # load all matrices
        self.indices = [i for i in range(2937)]

        self.labels = self.read_labels()

        self.sensor_data = self.read_sensor()
        self.image_data = 0 #np.random.rand(2937, 4, 2, 2, 1) # TODO PHILLIP load the correct dataset(based on window_size)

    def __len__(self) -> int:
        # amount of total samples / windows / whatever we train
        if (len(self.sensor_data) != len(self.image_data)):
            raise RuntimeError("Sensor data and video data have different sizes")
            return -1
        return len(self.labels)

    def __getitem__(self, index):
        # TODO maybe do a check before returning
        id = self.indices[index]

        if isinstance(index, int):
            images = np.reshape(np.load('./processed_dataset/video/window_' + str(self.window_size) + '/window_' + str(id) + ".npy"), (1,self.window_size, 224,224, 3))
            return (self.sensor_data[index], images , self.labels[index])

        else:
            length = len(id)-1
            images = np.reshape(np.load('./processed_dataset/video/window_' + str(self.window_size) + '/window_' + str(id[0]) + ".npy"), (1,self.window_size, 224,224, 3))


        for i in range(length):
            images = np.concatenate((images,np.reshape(np.load('./processed_dataset/video/window_' + str(self.window_size) + '/window_' + str(id[0]) + ".npy"), (1,self.window_size, 224,224, 3))), axis = 0) # (batch size, window_size, 224,224,3)

        return (self.sensor_data[index], images , self.labels[index])

    def read_sensor(self):
        dat_dir = './processed_dataset/sensor/dat/window_' + str(self.window_size)
        # get shape
        files = os.listdir(dat_dir)
        # read data
        file = open(dat_dir + "/" + files[1], 'rb')
        shapes = pickle.load(file)
        return np.memmap(dat_dir + "/" + files[2], dtype='float32', mode='r', shape=shapes['sensor'])
        
    def read_labels(self):
        dat_dir = './processed_dataset/sensor/dat/window_' + str(self.window_size)
        # get shape
        files = os.listdir(dat_dir)
        # read data
        file = open(dat_dir + "/" + files[1], 'rb')
        shapes = pickle.load(file)
        return np.memmap(dat_dir + "/" + files[0], dtype='int', mode='r', shape=shapes['labels'])


if __name__ == "__main__":
    # TODO: perform our tests
    d = Dataset()
    sensor, image, label = d[0:8]
    print(sensor.shape)
    print(image.shape)
    print(label.shape)

