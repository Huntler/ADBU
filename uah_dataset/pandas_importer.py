import os
from typing import List


class UAHDataset:
    def __init__(self) -> None:
        # load dataset versions available and choose the latest one
        self.__root_dir = "./uah_dataset/"
        self.__latest = self.latest

    @property
    def versions(self) -> List[str]:
        versions = []

        # make sure only folders named with "UAH-DRIVESET" are in the list of versions
        for path in os.listdir(self.__root_dir):
            if len(path) > 4 and path[0:12] == "UAH-DRIVESET":
                versions.append(path)

        if len(versions) == 0:
            raise RuntimeError(
                "Ensure to add a UAH-DRIVESET-v* to the 'uah_dataset' folder. " +
                "For more details, check the 'README.md' file located at './uah_dataset/'.")

        return versions

    @property
    def latest(self) -> str:
        # the latest version can be found by maximizing the list of versions
        return max(self.versions)
    
    @property
    def drivers(self) -> List[str]:
        recordings = []

        # make sure only folders named with "D" are in the list of drivers
        for subpath in os.listdir(f"{self.__root_dir}/{self.__latest}"):
            if len(subpath) == 2 and subpath[0] == "D":
                recordings.append(subpath)
        
        return recordings
