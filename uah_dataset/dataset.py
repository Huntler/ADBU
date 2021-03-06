import torch
import numpy as np
import os
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self, window_size: int) -> None:
        super().__init__()

        #provide with window size of data you want to load
        self.window_size = window_size

        # load all matrices
        self.indices = [i for i in range(2937)]

        self.labels = self.read_labels()

        self.sensor_data = self.read_sensor()
        self.image_data = 0 #np.random.rand(2937, 4, 2, 2, 1) # TODO PHILLIP load the correct dataset(based on window_size)

    def __len__(self) -> int:
        # amount of total samples / windows / whatever we train
        # if (len(self.sensor_data) != len(self.image_data)):
        #     raise RuntimeError("Sensor data and video data have different sizes")
        #     return -1
        return len(self.labels)

    def __getitem__(self, index):
        id = self.indices[index]
        # we need the class number, not onehot encoded to use cross-entropy as loss function
        labels = self.labels[index].astype(np.float32) #[[index]] to retain dimension
        labels = np.argmax(labels)
        labels = np.array(labels)

        if isinstance(index, int):
            images = np.reshape(np.load('./uah_dataset/processed_dataset/video/window_' + str(self.window_size) + '/window_' + str(id) + ".npy"), (self.window_size, 224,224, 3))
            images = images.astype(np.float32) / 255
            images = np.array(images)

            sensor = self.sensor_data[index]
            sensor = np.array(sensor)
            return (sensor, images , labels) # [[index]] to retain dims | [np.newaxis,...] for images

        else:
            length = len(id)-1
            images = np.reshape(np.load('./uah_dataset/processed_dataset/video/window_' + str(self.window_size) + '/window_' + str(id[0]) + ".npy"), (self.window_size, 224,224, 3))[np.newaxis,...]
        
        for i in range(length):
            images = np.concatenate((images,np.reshape(np.load('./uah_dataset/processed_dataset/video/window_' + str(self.window_size) + '/window_' + str(id[0]) + ".npy"), (self.window_size, 224,224, 3))[np.newaxis,...]), axis = 0) # (batch size, window_size, 224,224,3)
            images = images.astype(np.float32) / 255
            images = np.array(images)

        sensor = self.sensor_data[index]
        sensor = np.array(sensor)
        return (sensor, images, labels)

    def read_sensor(self):
        dat_dir = './uah_dataset/processed_dataset/sensor/dat/window_' + str(self.window_size)
        # get shape
        files = os.listdir(dat_dir)
        sensor_file = None
        shapes = None
        for file in files:
            if "train" in file:
                sensor_file = file
            if ".txt" in file:
                f = open(dat_dir + "/" + file, 'rb') 
                shapes = pickle.load(f)

        # read data
        return np.memmap(dat_dir + "/" + sensor_file, dtype='float32', mode='r', shape=shapes['sensor'])
        
    def read_labels(self):
        dat_dir = './uah_dataset/processed_dataset/sensor/dat/window_' + str(self.window_size)

        files = os.listdir(dat_dir)
        label_file = None
        shapes = None
        for file in files:
            if "labels" in file:
                label_file = file
            if ".txt" in file:
                f = open(dat_dir + "/" + file, 'rb') 
                shapes = pickle.load(f)

        return np.memmap(dat_dir + "/" + label_file, dtype='int', mode='r', shape=shapes['labels'])


if __name__ == "__main__":
    # TODO: perform our tests
    d = Dataset(window_size=30)
    sensor, image, label = d[0:5]
    print(sensor.shape)
    print(image.shape)
    print(label.shape)
    for X in d:
        sensor, image, y = X
        y = y
        print(sensor[0], image.shape, np.argmax(y))
