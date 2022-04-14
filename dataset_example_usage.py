import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


from uah_dataset.pandas_importer import UAHDataset

# create the dataset
dataset = UAHDataset()
print("Dataset:", dataset.latest)
print("Drivers:", dataset.drivers)


#test



'''# read one specific driver
print("Info of driver D1")
road_type_dict = dataset.dataframe_by_driver("D1", skip_missing_headers=True, suppress_warings=True)
roads = [_ for _ in road_type_dict.keys()]
print("Roads:\t", roads)
print("# Roads:", sum([len(road_type_dict[_]) for _ in roads]))

# read all drivers
# note: this should work, but we loose information about when a time series resets. 
# This could prevent learning later on.
print("Info of all drivers")'''
road_type_dict = dataset.dataframe(skip_missing_headers=True, suppress_warings=True)
'''roads = [_ for _ in road_type_dict.keys()]
print("# Roads:", sum([len(road_type_dict[_]) for _ in roads]))

for r, rd in road_type_dict.items():
    print(f"\t{r} - {len(rd)}")
    for l, ld in rd.items():
        print(f"\t\t{l} - {len(ld)}")

# process one specific dataframe'''
array = []
for road_type in road_type_dict.items():
    print(1)

    skip = True
    for route in road_type[1].items():
        mean = np.array(route[1].mean())
        array.append(mean)

#print(road_type_dict["SECONDARY"]["NORMAL"][['Altitude', 'Longitude', 'Latitude']].head(5))
#'Altitude', 'Longitude', 'Latitude'

fused_data = np.array(array)
fused_data[np.isnan(fused_data)]=0
kmeans = KMeans(n_clusters=2, random_state=0).fit(fused_data)


print(kmeans.labels_)
