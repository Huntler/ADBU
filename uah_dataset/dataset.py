import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        # load all matrices

    def __len__(self) -> int:
        # amount of total samples / windows / whatever we train
        return 0

    def __getitem__(self, index):
        #
        return None


if __name__ == "__main__":
    # TODO: perform our tests
    d = Dataset()

    print(len(d))
    (sensor_data, image_data), label = d[0]
