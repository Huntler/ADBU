import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf


from uah_dataset.pandas_importer import UAHDataset


# TODO: find missing headers: have a look into their data reader again
# EFFORT: 2h (at most)

# TODO: sample frame of video corresponding to sensor data depending on time step
# EFFORT: 2h (depending on libraries i can use)

# TODO: apply some preprocessing on image data (image normalization, contrast sharpening, …)
# EFFORT depends on the amount of preprocessing

# TODO: save this form of dataset created by us and save it (using pandas and/or our own datastructure)

# to extract video data (run this one time only to set up your dataset correctly)
# UAHDataset(generate_video_frames=True)

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

print(fused_data.shape)

# define model
'''model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')'''


'''Then we will extract the inner layer after training'''


'''We will finaly cluster based on the representation'''
print(kmeans.labels_)
