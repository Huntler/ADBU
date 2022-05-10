from argparse import ArgumentError
from typing import Dict, List, Tuple
import os
import pandas


headers = {
    "RAW_ACCELEROMETERS": ["time", "Activation bool (1 if speed>50Km/h)", "X acceleration (Gs)", "Y acceleration (Gs)", "Z acceleration (Gs)",
                           "X accel filtered by KF (Gs)", "Y accel filtered by KF (Gs)", "Z accel filtered by KF (Gs)", "Roll (degrees)", "Pitch (degrees)",
                           "Yaw (degrees)"],
    "RAW_GPS": ["time", "Speed (Km/h)", "Latitude", "Longitude", "Altitude", "Vertical accuracy", "Horizontal accuracy",
                "Course (degrees)", "Difcourse: course variation", "Position state [internal val]", "Lanex dist state [internal val]",
                "Lanex history [internal val]", "Unkown"],
    "PROC_LANE_DETECTION": ["time", "Car pos. from lane center (meters)", "Phi", "Road width (meters)", "State of lane estimator"],
    "PROC_VEHICLE_DETECTION": ["time", "Distance to ahead vehicle (meters)", "Impact time to ahead vehicle (secs.)", "Detected # of vehicles",
                               "Gps speed (Km/h) [redundant val]"],
    "PROC_OPENSTREETMAP_DATA": ["time", "Current road maxspeed", "Maxspeed reliability [Flag]", "Road type [graph not available]", "# of lanes in road",
                                "Estimated current lane", "Latitude used to query OSM", "Longitude used to query OSM", "Delay answer OSM query (seconds)", "Speed (Km/h) [redundant val]"]
}


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

    def dataframe(self, skip_missing_headers: bool = False, suppress_warings: bool = False) -> Dict:
        complete = {}
        for driver in self.drivers:
            road_type_dict = self.dataframe_by_driver(driver, skip_missing_headers, suppress_warings)
            for road_type, label_dict in road_type_dict.items():
                c = complete.get(road_type, {})
                for label, data in label_dict.items():
                    l = c.get(label, pandas.DataFrame())
                    if l.empty:
                        l = data
                    else:
                        l = pandas.concat([l, data])

                    c[label] = l
                complete[road_type] = c

        return complete             

    def dataframe_by_driver(self, driver: str, skip_missing_headers: bool = False, suppress_warings: bool = False) -> Dict:
        """This method loads the recordings of a provided driver into a pandas dataframe.

        Args:
            driver (str): The driver of the recordings. Has to match the folder's name e.g. D1
            skip_missing_headers (bool, optional): If headers of a text-file is missing, then this 
                file will be skipped and is not included in the final dataframe. Defaults to False.
            supress_warnings (bool, optional): Supresses warnings which may occur.

        Raises:
            ArgumentError: Occurs if the provided driver is not availabe.
            RuntimeError: Can occur if the header is missing or if the header was set to a wrong value.

        Returns:
            Tuple[Dict]: Returns a dict (key: road_type) with the keys as a dataframe.
        """
        if driver not in self.drivers:
            raise ArgumentError(
                f"There is not recording for the driver {driver} in the current dataset {self.__latest}")

        folder = f"{self.__root_dir}/{self.latest}/{driver}"
        road_types = ["SECONDARY", "MOTORWAY"]

        road_type_dict = {}
        for rec in os.listdir(folder):
            time_stamp, distance, _, behaviour, road_type = rec.split("-")

            merged_data = pandas.DataFrame()
            for file in os.listdir(f"{folder}/{rec}"):
                if file[-4:] != ".txt" or file == "SEMANTIC_FINAL.txt":
                    continue
                    #TODO why we do not consider this file?

                # read the dataset file into a pandas dataframe
                try:
                    data = pandas.read_csv(
                        f"{folder}/{rec}/{file}", sep=" ", header=None)
                except pandas.errors.EmptyDataError:
                    pass

                # check if file has a registered header
                if file[:-4] not in headers.keys():
                    if skip_missing_headers:
                        if not suppress_warings:
                            print(f"WARNING: Skipped {file[:-4]} due to missing header.")
                        continue

                    raise RuntimeError(
                        f"No header specified for dataset file {file}.")

                # check if the header was correct
                if len(data.columns) != len(headers[file[:-4]]):
                    num_nans = data.iloc[:, -1].isnull().sum()
                    if num_nans != len(data.iloc[:, -1]):
                        raise RuntimeError(
                            f"Headers specified for dataset file {file} have the wrong length. " +
                            f"Expected {len(data.columns)} but found {len(headers[file[:-4]])}.")

                    # drop last column due to wrong parsing
                    data.drop(
                        data.columns[[len(data.columns) - 1]], axis=1, inplace=True)

                data.columns = headers[file[:-4]]

                # merge the dataset files into the dataframe
                if merged_data.empty:
                    merged_data = data
                    continue

                merged_data = merged_data.merge(
                    data, how="left", on="time")  # , on="time")

            # store the merged data corresponding to several keys
            label_dict = road_type_dict.get(road_type, {})
            data = label_dict.get(behaviour, pandas.DataFrame())
            if data.empty:
                data = merged_data
            else:
                data = pandas.concat([data, merged_data])

            # replace values, which are NaN using interpolation of surroundings
            data.interpolate(method="linear", inplace=True)
            data.fillna(method='bfill', inplace=True)

            label_dict[behaviour] = data
            road_type_dict[road_type] = label_dict

        return road_type_dict
