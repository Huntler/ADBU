from uah_dataset.pandas_importer import UAHDataset

# TODO: create entry for each time step: problem is "how to fill missing values?"
# approach 1: simply repeat previous values
# approach 2: calculate some kind of regression or use splines
# approach X: ?
# EFFORT: 6h? (depending on approach and existing data)
# Florene

# TODO: find missing headers: have a look into their data reader again
# EFFORT: 2h (at most)
# Jonas

# TODO: sample frame of video corresponding to sensor data depending on time step
# EFFORT: 2h (depending on libraries i can use)
# Jonas or Florene

# TODO: apply some preprocessing on image data (image normalization, contrast sharpening, …)
# EFFORT depends on the amount of preprocessing
# Jonas

# TODO: save this form of dataset created by us and save it (using pandas and/or our own datastructure)
# Florene or Jonas

# create the dataset
dataset = UAHDataset()
print("Dataset:", dataset.latest)
print("Drivers:", dataset.drivers)

# read one specific driver
print("Info of driver D1")
road_type_dict = dataset.dataframe_by_driver("D1", skip_missing_headers=True, suppress_warings=True)
roads = [_ for _ in road_type_dict.keys()]
print("Roads:\t", roads)
print("# Roads:", sum([len(road_type_dict[_]) for _ in roads]))

# read all drivers
# note: this should work, but we loose information about when a time series resets. 
# This could prevent learning later on.
print("Info of all drivers")
road_type_dict = dataset.dataframe(skip_missing_headers=True, suppress_warings=True)
roads = [_ for _ in road_type_dict.keys()]
print("# Roads:", sum([len(road_type_dict[_]) for _ in roads]))

for r, rd in road_type_dict.items():
    print(f"\t{r} - {len(rd)}")
    for l, ld in rd.items():
        print(f"\t\t{l} - {len(ld)}")

# process one specific dataframe
print(road_type_dict["SECONDARY"]["NORMAL"].head())
