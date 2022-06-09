import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, d_type: str = "train") -> None:
        super().__init__()

        # get shapes of files
        """with open('../train.npy', 'rb') as f:
            #major, minor = np.lib.format.read_magic(f)
            sensor_shape, _, sensor_dtype = np.lib.format.read_array_header_1_0(f)
        with open('../labels.npy', 'rb') as f:
            #major, minor = np.lib.format.read_magic(f)
            image_shape, _, image_dtype = np.lib.format.read_array_header_1_0(f)
            
        #load all matrices
        self.sensor_data = np.memmap('../train.npy', dtype=sensor_dtype, mode='w+', shape=sensor_shape)
        self.image_data = np.random.rand(2937, 400, 2, 2, 1)
        self.labels = np.memmap('../labels.npy', dtype=image_dtype, mode='w+', shape=image_shape)"""

        # load all matrices
        self.sensor_data = np.memmap('./uah_dataset/processed_dataset/train_processed.dat', dtype='float32', mode='r', shape=(2937, 400, 36))
        self.image_data = np.random.rand(2937, 400, 2, 2, 3) # 224, 224, 3
        self.labels = np.memmap('./uah_dataset/processed_dataset/labels_processed.dat', dtype='int', mode='r', shape=(2937, 3))

        if d_type == "train":
            # TODO: training data
            pass
        elif d_type == "test":
            # TODO: testing data
            pass
        else:
            raise RuntimeError("Unsupported dataset type: " + d_type)

    def __len__(self) -> int:
        # amount of total samples / windows / whatever we train
        if (len(self.sensor_data) != len(self.image_data)):
            raise RuntimeError("Sensor data and video data have different sizes")

        return len(self.labels)

    def __getitem__(self, index):
        # TODO maybe do a check before returning
        #print(self.labels)
        return ((self.sensor_data[index], self.image_data[index]), self.labels[index])


if __name__ == "__main__":
    # perform some tests
    d = Dataset()

    print(len(d))
    (sensor_data, image_data), label = d[...]
    print(np.max(sensor_data))
    print(image_data)
    print(label)
    #print(label)

