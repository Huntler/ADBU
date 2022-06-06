import numpy as np
import imageio
import os
import pandas as pd





def windowing(dictionary : dict ,rows_per_minute : int = 360, initial_threshold : int = 60, increment : int = 10) -> dict:
    """
    Creates windows, for every
    :param dictionary: nested dic road types -> mood -> dataframe
    :param rows_per_minute: number of rows that on averag consistute one minute
    :param initial_threshold: timestamp when we start to approve the windows
    :param increment: timestamp difference between end of adjacent windows
    :return: dictionary: road type -> mood -> window_index -> dataframe
    """

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
            for window in mood_df.rolling(window = rows_per_minute):
                if window.iloc[-1, 0] < window.iloc[0, 0]:  # meaning we have finished one driver trip, as the nnext df values are lower than the previous
                    t = initial_threshold
                elif int(window.iloc[-1, 0]) > t:
                    if len(window) == rows_per_minute:
                        windowed[i] = window
                        i += 1
                        window_number += 1
                        time = (window.iloc[-1, 0]-window.iloc[0,0])+1
                        window_time += time
                        time_difference.append(time)
                    t += increment                     #creates 10 second windows

            dictionary[road][mood] = windowed

    # print(f"Average window timelapse: {window_time/window_number}")
    # print(f"Number of windows: {window_number}")
    # plt.hist(time_difference, bins = 120)
    # plt.title(f"Window timelapse for {rows_per_minute} data points.")
    # plt.xlabel("Seconds")
    # plt.show()
    return dictionary

def add_pointers_to_window(df, roadtype: str, mood : str)->tuple:
    """
    This function will createa  list of pointers, that will match the time stamp on the dataframe
    :param dataframe, the roadtype and mood for the corresponding dataframe
    :return the dataframe and the pointers that correspond to the images as a list
    """

    # fps  = 29.97 #constant among all videos
    fps = 1 #test
    fps_to_model = 1 #How many frames to feed to the model per second
    initial_time = df.iloc[0]['TimeStamp']
    first_frame = int(initial_time * fps)
    final_time = df.iloc[-1]['TimeStamp']
    final_frame = int(final_time * fps)
    time_span = final_time - initial_time  #is the time between the first and last data point in this frame
    driver = df.iloc[0]['Driver']       #gets the driver name
    folder = f"uah_dataset/UAH-DRIVESET-v1/{driver}"
    name = mood + "-" + roadtype
    frames = np.empty((1,540,960), dtype = "float64")
    frame_pointers = []
    for file in os.listdir(folder):
        if name in file:
            frames_path = folder +"/" +file + "/frames"   #Finds the folder where the frames are stored as png

    for i in range(first_frame, final_frame+1,int(fps/fps_to_model)):  #the frames number to save, the step is the reciprocal of the desired fps, times the og fps
        path_to_frame = f"{frames_path}/frame{i:06d}.png"
        frame_pointers.append(path_to_frame)


    return (df, frame_pointers)

def dict_with_all_frames_pointed(pointers):
    """"This function returns a dictionary, with the pointers as keys
    and the image being pointed to as value
    :param list of pointers lists
    :return dictionary pointer as key, image as value
    """

    if len(pointers) == 0:
        return

    dictionary = {}
    i = 0
    total = len(pointers)
    for pointer in pointers:  #for each list within the list of pointers
        i += 1
        percent = float("{0:.2f}".format(i / total * 100))
        filledLength = int(percent)
        bar = '█' * filledLength + '-' * (100 - filledLength)
        print(f'\r progress |{bar}| {percent}% Loading images to dictionary', end = ' ')


        if len(pointer) == 0:   #if there is no pointer in this list continue
            continue

        for point in pointer:  #for each point in the list of pointers
            if point in dictionary:
                continue #no need to reload the image

            dictionary[point] = imageio.imread(point)[:,:,0]


    return dictionary



def read(path_to_uah_folder: str = f"{os.path.dirname(__file__)}/uah_dataset/UAH-DRIVESET-v1/"):
    """
    Mainly copied from the original reader
    :param path_to_uah_folder:
    :return: Online semantics, in nested dictionary ->road type -> mood -> dataframe for all driver on this mood and road type
    """
    root_dir = "./uah_dataset/"
    latest = "UAH-DRIVESET-v1/"
    drivers  = ['D1','D2','D3','D4','D5','D6']
    roads = ["MOTORWAY", "SECONDARY"]

    headers = ["TimeStamp", "Latitude" , " Longitude", "Total" , "Accel", "Braking", "Turning", "Weaving",
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

