import numpy as np
import os
import pandas as pd
import imageio

root_dir = "./uah_dataset"
latest = "UAH-DRIVESET-v1"
drivers  = ['D1','D2','D3','D4','D5','D6']
roads = ["MOTORWAY", "SECONDARY"]

headers = ["TimeStamp", "Latitude" , " Longitude", "Total" , "Accel", "Braking", "Turning", "Weaving",
            "Drifting", "Oversspeed", "Carfollow", "Normal", "Drowsy", "Aggressive", "Unknown",
            "Total_last_minute", "Accel_last_minute", "Braking_last_minute" , "Turning_last_minute",
            "Weaving_last_minute","Drifting_last_minute", "Oversspeed_last_minute", "Carfollow_last_minute",
            "Normal_last_minute", "Drowsy_last_minute", "Aggressive_last_minute", "Unknown_last_minute"
            ]

def read():
    online_semantics = {}
    for driver in drivers:
        folder = f"{root_dir}/{latest}/{driver}"
        driver_semantics = {}
        for direc in os.listdir(folder):
            splitted_string = direc.split('-')

            scoresFileName = folder + '/' + direc + '/' + 'SEMANTIC_ONLINE.txt'
            scoresData = np.genfromtxt(scoresFileName, dtype=np.float64, delimiter=' ')

            df = pd.DataFrame(scoresData, columns = headers)
            videoName = folder + "/" + direc + "/" + direc + ".mp4"
            if  os.path.isfile(videoName):
                videoReader = imageio.get_reader(videoName, 'ffmpeg')
                video_meta = videoReader.get_meta_data()   #returns dict
                fps = video_meta['fps']


            if splitted_string[4] in driver_semantics:
                driver_semantics[splitted_string[4]].update({splitted_string[3]: df})
            else:
                driver_semantics[splitted_string[4]] = {splitted_string[3]: df}

        online_semantics[driver] = driver_semantics

    return online_semantics

