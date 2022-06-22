import pandas as pd
import numpy as np
import copy
import os

from uah_dataset.pandas_importer import UAHDataset
from uah_dataset.image_process import add_pointers_to_window, dict_with_all_frames_pointed, video_to_frames, create_windowed_frames
import shutil
import pickle
from datetime import datetime
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


def windowing(dictionary : dict ,rows_per_minute : int = 360, initial_threshold : int = 60, increment : int = 10) -> dict:
    """
    Creates windows, for every driver's trip.
    :param dictionary: nested dic road types -> mood -> dataframe
    :param rows_per_minute: number of rows that on average constitute one minute
    :param initial_threshold: timestamp when we start to approve the windows
    :param increment: timestamp difference between end of adjacent windows
    :return: dictionary: road type -> mood -> window_index -> dataframe
    """

    windowed_dic = copy.deepcopy(dictionary)
    window_number = 0
    window_time = 0
    time_difference = []

    for road, road_dic in dictionary.items():
        i = 0
        print(f"______________________Road: {road} _________________________________")
        for mood, mood_df in road_dic.items():
            windowed = {}
            t = initial_threshold                                    #first window ends at a point where more than t seconds have passed
            print(f"Mood: {mood}")
            '''for window in mood_df.rolling(window = rows_per_minute, min_periods = rows_per_minute):
                if window.shape != (2,41):
                    print('something went wrong')
                if window.iloc[-1, 0] < window.iloc[0, 0]:  # meaning we have finished one driver trip, as the nnext df values are lower than the previous
                    #TODO keep an eye om this
                    t = initial_threshold
                elif int(window.iloc[-1, 0]) > t:
                    #TODO time does not begin from zero
                    windowed[i] = window
                    i += 1
                    t += increment                     #creates 10 second windows
                    window_number += 1
                    time = (window.iloc[-1, 0]-window.iloc[0,0])+1
                    window_time += time
                    time_difference.append(time)'''

            for j in mood_df.index:
                try:
                    window = mood_df[j:j+rows_per_minute]
                except Exception:
                    pass
                if window.shape != (rows_per_minute,41):
                    print('pass')
                    pass
                if window.iloc[-1, 0] < window.iloc[0, 0]:  # meaning we have finished one driver trip, as the nnext df values are lower than the previous
                    #TODO keep an eye om this
                    t = initial_threshold
                elif int(window.iloc[-1, 0]) > t:
                    #TODO time does not begin from zero
                    windowed[i] = window
                    i += 1
                    t += increment                     #creates 10 second windows
                    window_number += 1
                    time = (window.iloc[-1, 0]-window.iloc[0,0])+1
                    window_time += time
                    time_difference.append(time)
            windowed_dic[road][mood] = windowed
    print(f"Number of windows: {window_number}")

    return windowed_dic



def read(path_to_uah_folder: str = f"{os.path.dirname(__file__)}/uah_dataset/UAH-DRIVESET-v1/"):
    """
    Note: we ended up not using this loading system
    Mainly copied from the original reader.
    :param path_to_uah_folder:
    :return: Online semantics, in nested dictionary ->road type -> mood -> dataframe for all driver on this mood and road type
    """
    root_dir = "./uah_dataset/"
    latest = "UAH-DRIVESET-v1/"
    drivers = ['D1','D2','D3','D4','D5','D6']
    roads = ["MOTORWAY", "SECONDARY"]

    headers = ["time", "Latitude" , " Longitude", "Total" , "Accel", "Braking", "Turning", "Weaving",
                "Drifting", "Oversspeed", "Carfollow", "Normal", "Drowsy", "Aggressive", "Unknown",
                "Total_last_minute", "Accel_last_minute", "Braking_last_minute" , "Turning_last_minute",
                "Weaving_last_minute","Drifting_last_minute", "Oversspeed_last_minute", "Carfollow_last_minute",
                "Normal_last_minute", "Drowsy_last_minute", "Aggressive_last_minute", "Unknown_last_minute"
                ]
    online_semantics = {}
    for driver in drivers:
        folder = f"{path_to_uah_folder}{driver}"
        for direc in os.listdir(folder):
            splitted_string = direc.split('-')
            road = splitted_string[-1]
            mood = splitted_string[-2]
            scoresFileName = folder + '/' + direc + '/' + 'SEMANTIC_ONLINE.txt'
            scoresData = np.genfromtxt(scoresFileName, dtype=np.float64, delimiter=' ')
            df = pd.DataFrame(scoresData, columns = headers)
            df['Driver'] = driver
            if road in online_semantics:
                if mood in online_semantics[road]:
                    online_semantics[road][mood] = pd.concat([online_semantics[road][mood], df])
                else:
                    online_semantics[road][mood] = df
            else:
                online_semantics.update({road : {mood : df}})


    return online_semantics

def reshaping_to_numpy(dataf : pd.DataFrame, window_size):
    """"
    This function will convert the data from the dataframes in the dictionary into a numpy array.
    :param dataf dictionary: road type -> mood -> window_index -> dataframe
    :param window_size int: the size of the window
    :return numpy array (number of windows, window_size, number_of_features)
    """

    feature_size = list(list(list(dataf.values())[0].values())[0].values())[0].shape[1] #TODO change that again
    train = np.empty([0,window_size, feature_size])
    labels = np.empty([0,3], dtype=int)
    for road, road_dic in dataf.items():
        for mood, mood_df in road_dic.items():
            if "NORMAL" in mood:
                label = np.array([1,0,0], dtype=int)
            elif "AGGRESSIVE" in mood:
                label = np.array([0,1,0], dtype=int)
            elif "DROWSY" in mood:
                label = np.array([0,0,1], dtype=int)
            else:
                raise RuntimeError(mood , " does not correspond to any existing labels")

            for i in mood_df:
                train = np.concatenate((train, mood_df[i].values[np.newaxis,...]), axis=0)
                labels = np.concatenate((labels, label[np.newaxis,...]), axis=0)
            print('Iteration passed')
    print(train.shape)
    print(labels.shape)

    return train, labels



def sensor_data_prepare(window_size):
    """"
    This function will prepare the sensor data. It takes in the desired size of the window,
    it then loads the data, and calls the window function. It then transforms the windows into numpy,
    performs some preprocessing and saves the values as a memory map.
    :param window_size int: the desired size of the window
    :return indexing This is permutation with length equal to the number of windows, we use this to match the video
    windows to the correct sensor windows
    :return n_samples int: is the number of windows
    :return dictionary Road type -> mood -> window_index -> dataframe
    """

    now = datetime.now()  # current date and time
    # Read data from files and store to panda frames
    dataset = UAHDataset()
    road_type_dict = dataset.dataframe(skip_missing_headers=True, suppress_warings=True)
    # Windowing the dataset
    windowed_dic = copy.deepcopy(road_type_dict)
    online_semantic = windowing(windowed_dic, rows_per_minute=window_size)
    pass
    # Reshaping to numpy
    train, labels = reshaping_to_numpy(online_semantic, window_size)

    n_samples = len(train)
    indexing = np.random.permutation(n_samples)

    (train,labels) = (train[indexing],labels[indexing])

    npy_new_dir = './uah_dataset/processed_dataset/sensor'
    if not os.path.exists(npy_new_dir):
        os.mkdir(npy_new_dir)
    npy_new_dir = npy_new_dir + '/npy'
    if not os.path.exists(npy_new_dir):
        os.mkdir(npy_new_dir)
    npy_new_dir = npy_new_dir + '/window_' + str(window_size)
    if os.path.exists(npy_new_dir):
        shutil.rmtree(npy_new_dir)
    os.mkdir(npy_new_dir)
    np.save(npy_new_dir + '/train_' + now.strftime("%m_%d_%Y-%H_%M_%S"), train)
    np.save(npy_new_dir + '/labels_' + now.strftime("%m_%d_%Y-%H_%M_%S"), labels)



    parent_dir = './uah_dataset/processed_dataset/sensor/npy/window_' + str(window_size)
    train_path = labels_path = None
    for file in os.listdir(parent_dir):
        if "train" in file:
            train_path = file
        else:
            labels_path = file
            
    # read data
    train = np.load(parent_dir + "/" + train_path, allow_pickle=True)
    labels = np.load(parent_dir + "/" + labels_path, allow_pickle=True)
    train_processed = train

    # train = np.load("train.npy",allow_pickle=True)
    # labels = np.load("labels.npy",allow_pickle=True)



    # 0:(OUT) Timestamp
    # 1: Car position form lane center (m)
    # 2: Phi
    # 3:(OUT) Road width
    # 4: State of lane estimator
    # 5: Current road max speed
    # 6:(OUT) Max speed reliability
    # 7:(OUT) Road type
    # 8: #lanes in road
    # 9: estimated current lane
    # 10: Latitude used to query OSM
    # 11:(OUT) Longitude used to query OSM
    # 12:(OUT) Delay answer OSM Query (s)
    # 13:(OUT) Speed (kmh)
    # 14: Distance to ahead vehicle
    # 15: Impact to ahead vehicle (s)
    # 16: Detectet # of vehicles
    # 17: GPS speed (kmh)
    # 18: Activation Boolean (speed>50kmh)
    # 19: X acceleration (Gs)
    # 20: Y acceleration(Gs)
    # 21:(OUT) Z acceleration(Gs)
    # 22: X acceleration (Gs)(Kernel Filter)
    # 23: Y acceleration(Gs)(Kernel Filter)
    # 24: Z acceleration(Gs)(Kernel Filter)
    # 25: Roll
    # 26:(OUT) Pitch
    # 27:(OUT) Yaw
    # 28:(OUT)Speed
    # 29:(OUT) Latitude
    # 30:(OUT) Longitude
    # 31:(OUT) Altitude
    # 32: Vertical Accuracy
    # 33: Horizontal accuracy
    # 34: Course
    # 35: Difcourse: course variation
    # 36:(OUT) Position state
    # 37:(OUT) Lanex dist state
    # 38:(OUT) Lanex history
    # 39:(OUT) Unknown
    # 40:(OUT) Driver


    idx_OUT_columns = [0, 3, 6, 7, 11, 12, 13, 21,26,27,28, 29, 30, 31, 36, 37, 38, 39, 40]
    idx_IN_columns = [i for i in range(np.shape(train_processed)[2]) if i not in idx_OUT_columns]
    extractedData = train_processed[:, :, idx_IN_columns]

    extractedData = extractedData.astype(np.float32)
    #Remove NaN values
    extractedData = np.nan_to_num(extractedData, copy=True, nan=0.0, posinf=None, neginf=None)

    #Normalize train by feautures (column)
    scalers = {}
    for i in range(extractedData.shape[2]):
        scalers[i] = StandardScaler() if scaler == "standard" else MinMaxScaler()
        extractedData[:, :, i] = scalers[i].fit_transform(extractedData[:, :, i]) 

    #for test
    '''for i in range(extractedData.shape[2]):
        extractedData[:, :, i] = scalers[i].transform(extractedData[:, :, i])'''

    """#Normalize train by feautures (column)
    # FIXME: normalize over the whole dataset
    for j in range(len(extractedData)):
        df1=pd.DataFrame(extractedData[j])
        for i in range(0, len(idx_IN_columns)):
            df1[i] = (df1[i] / (df1[i].abs().max()+0.01)).astype(float)

        extractedData[j]=df1.to_numpy()"""

    # save data to .dat format
    dat_new_dir = './uah_dataset/processed_dataset/sensor'
    if not os.path.exists(dat_new_dir):
        os.mkdir(dat_new_dir)
    dat_new_dir = dat_new_dir + '/dat'
    if not os.path.exists(dat_new_dir):
        os.mkdir(dat_new_dir)
    dat_new_dir = dat_new_dir + '/window_' + str(window_size)

    if os.path.exists(dat_new_dir):
        shutil.rmtree(dat_new_dir)
    os.mkdir(dat_new_dir)

    # save train data
    fp = np.memmap(dat_new_dir + '/train_' + now.strftime("%m_%d_%Y-%H_%M_%S") + ".dat", dtype='float32', mode='w+',
                   shape=extractedData.shape)
    fp[:] = extractedData[:]
    fp.flush()
    del fp

    print(extractedData.shape)



    # save label data
    labels_processed = labels
    dp = np.memmap(dat_new_dir + '/labels_' + now.strftime("%m_%d_%Y-%H_%M_%S") + ".dat", dtype='int', mode='w+',
                   shape=labels_processed.shape)
    dp[:] = labels_processed[:]
    dp.flush()

    del dp

    # save shape
    dict = {'sensor': extractedData.shape, 'labels': labels_processed.shape}
    file = open(dat_new_dir + '/shape.txt', 'wb')
    pickle.dump(dict, file)
    file.close()

    return indexing, n_samples, online_semantic

scaler = "minmax"
if __name__ == "__main__":
    # Create the parser
    dataset = UAHDataset()
    road_type_dict = dataset.dataframe(skip_missing_headers=True, suppress_warings=True)

    parser = argparse.ArgumentParser(description='Preprocessing stage')
    parser.add_argument('--window_size', type=int, help='window_size', required=True)
    parser.add_argument('--scaler', type=str, help='either minmax or standard', required=True)
    args = parser.parse_args()
    window_size = args.window_size
    scaler = args.scaler
    
    (indexing, n_samples, online_semantic) = sensor_data_prepare(window_size)
    fps = (window_size/60)
    video_to_frames(fps)
    create_windowed_frames(window_size, indexing, n_samples, online_semantic)




