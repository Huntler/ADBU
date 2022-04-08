from uah_dataset.pandas_importer import UAHDataset

# create the dataset
dataset = UAHDataset()
print("Dataset:", dataset.latest)
print("Drivers:", dataset.drivers)

road_type_dict, label_dict = dataset.dataframe("D1", skip_missing_headers=True, suppress_warings=True)
roads = [_ for _ in road_type_dict.keys()]
labels = [_ for _ in label_dict.keys()]
print("Roads:\t", roads)
print("Labels:\t", labels)

print("# Roads:", sum([len(road_type_dict[_]) for _ in roads]))
print("# Labels:", sum([len(label_dict[_]) for _ in labels]))
