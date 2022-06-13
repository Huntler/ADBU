import numpy as np
import os
import time
import imageio
import shutil







def add_pointers_to_window(df, roadtype: str, mood : str, fps : float = 1, window_size : int = 400)->tuple:
    """
    This function will createa  list of pointers, that will match the time stamp on the dataframe
    :param dataframe, the roadtype and mood for the corresponding dataframe
    :return the dataframe and the pointers that correspond to the images as a list
    """

    # fps  = 29.97 #constant among all videos
    fps_to_model = 1 #How many frames to feed to the model per second
    initial_time = df.iloc[0]['time']
    first_frame = int(initial_time * fps)
    final_time = df.iloc[-1]['time']
    final_frame = int((final_time+1) * fps)
    driver = df.iloc[0]['Driver']       #gets the driver name
    folder = f"uah_dataset/UAH-DRIVESET-v1/{driver}"
    name = mood + "-" + roadtype
    frame_pointers = []
    for file in os.listdir(folder):
        if name in file:
            frames_path = f"{folder}/{file}/frames_rgb_{fps:.0f}" #Finds the folder where the frames are stored as png
    last_catch = f"{frames_path}/frame{1:06d}.png"
    for i in range((final_frame-window_size), final_frame):  #the frames number to save, the step is the reciprocal of the desired fps, times the og fps
        path_to_frame = f"{frames_path}/frame{(i+1):06d}.png"
        if not os.path.isfile(path_to_frame):
            print(f"File not found: {path_to_frame}, appending last")
            frame_pointers.append(last_catch)
            continue

        last_catch = path_to_frame

        frame_pointers.append(last_catch)


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
    start_time = time.time()
    for pointer in pointers:  #for each list within the list of pointers
        i += 1
        percent = float("{0:.2f}".format(i / total * 100))
        filledLength = int(percent)
        bar = '█' * filledLength + '-' * (100 - filledLength)
        print(f'\rProgress |{bar}| {percent}% Loading images.', end = ' ')


        if len(pointer) == 0:   #if there is no pointer in this list continue
            continue

        for point in pointer:  #for each point in the list of pointers
            if point in dictionary:
                continue #no need to reload the image

            dictionary[point] = imageio.imread(point)#[:,:,0]

    print(f"\nTime taken to load: {(time.time() - start_time):.2f}s")
    return dictionary


def video_to_frames(fps : int = 1) -> None:
    """This method extracts all needed video frames and stores them as individual png frames
        Run this once to transform the data into a directory containing all the frames.
        One new directory per video

        Note: you need ffmpeg installed on your pc to be able to run this function

        :param number of fps to be saved to
        :return None

    """
    latest = "uah_dataset/UAH-DRIVESET-v1/"

    folder = f"{latest}"

    for driver in os.listdir(folder): #for all directories, i.e. all driver folders
        if os.path.isfile(f"{folder}/{driver}"):
            continue
        if not driver[0] == "D":
            continue

        folder_driver = folder + "/" + driver
        for rec in os.listdir(folder_driver):  #all recordings for every driver
            info = rec.split("-")
            if len(info) != 5:
                continue

            time_stamp, distance, _, behaviour, road_type = info

            # find the video file
            video_file = None
            for file in os.listdir(f"{folder_driver}/{rec}"):  #find the mp4 video file
                if file[-4:] == ".mp4":
                    video_file = file
                    break

            if video_file is None:
                raise RuntimeError(
                    f"Video file for driver {driver} with behaviour {behaviour} on road {road_type} does not exist.")


            # create a frame folder in which all frames are stored
            if not os.path.exists(f"{folder_driver}/{rec}/frames_rgb_{fps:.0f}"):
                os.mkdir(f"{folder_driver}/{rec}/frames_rgb_{fps:.0f}")
            else: #if it already exist, we can either:
                shutil.rmtree(f"{folder_driver}/{rec}/frames_rgb_{fps:.0f}")  #remove and create a new directory
                os.mkdir(f"{folder_driver}/{rec}/frames_rgb_{fps:.0f}")

                # raise RuntimeError("Frames were extracted, nothing to do.") #raise error

                print(f"File found {folder}/{rec}")  #skip the folder
                # continue

            os.system(f"ffmpeg -i {folder_driver}/{rec}/{video_file} -vf fps={fps},scale=224x224 {folder_driver}/{rec}/frames_rgb_{fps:.0f}/frame%6d.png")

    return None



def create_windowed_frames(window_size, indices, n_samples, online_semantic):
    """
    This function will create a numpy array (400,224,224), so that it can be called by the __getitem__ from the data set

    :param semantics: dictionaries of the dataframes
    :param index_list: list from 0 to n, shuffled together with telemetry data
    :return: None
    """

    fps  = window_size/60
    idx = 0
    path_to_save = f"uah_dataset/processed_dataset/video"
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    path_to_save = f"{path_to_save}/window_{window_size}"
    if os.path.exists(path_to_save):
        shutil.rmtree(path_to_save)

    os.mkdir(path_to_save)
    for road, road_dic in online_semantic.items():
        for mood, mood_dic in road_dic.items():
            list_pointers = []
            for df_index, df in mood_dic.items():
                list_pointers.append(add_pointers_to_window(df, road, mood, fps=fps, window_size=window_size)[1])
            dic = dict_with_all_frames_pointed(list_pointers)

            for i in range(len(list_pointers)):
                if not len(list_pointers[i]) == window_size:
                    print(f"Length list of pointers {len(list_pointers[i])}")
                    continue
                window_images = np.reshape(dic[list_pointers[i][0]], (1,224,224,3))
                for pointer in list_pointers[i][1:]:
                    window_images = np.concatenate((window_images, np.reshape(dic[pointer], (1,224,224,3))), axis = 0)


                filename = f"{path_to_save}/window_{indices[idx]}"
                np.save(filename, window_images)
                idx += 1





