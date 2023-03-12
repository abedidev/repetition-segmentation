import os
from os.path import exists

for exercise in range(5):
    for num_layers in range(1,4):
        path = "Ex"+str(exercise)+"_DMap/LSTM-CNN-Ex"+str(exercise)+"-BJ-L"+str(num_layers)+"/"
        video = 1
        while(exists(path + "Exercises/"+str(exercise)+"/videos/V"+str(video))):
            video += 1
        num_vids = video - 1
        
        for video in range(1, num_vids + 1):
            with open(path + "Exercises/"+str(exercise)+"/videos/V"+str(video)+"/Features.txt", "r") as f:
                num_frames = 0
                for line in f:
                    num_frames += 1
                dMap = ""
                with open(path + "Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video)+"_DM.csv", "r") as dm:
                    dMap = dm.readline()
                    dMapLen = len(dMap.split(","))
                    while(len(dMap.split(",")) < num_frames):
                        dMap = dMap.split("\n")[0]
                        dMap  = dMap.split(",")[0] + "," + dMap
                        dMap = dMap + "\n"
                    while(num_frames < len(dMap.split(","))):
                        dMap = dMap.split("\n")[0]
                        lst = dMap.split(",")
                        dMap = ""
                        for ele in lst[:-1]:
                            dMap += (ele + ",")
                        dMap = dMap[:-1]
                        dMap += "\n"
                with open(path + "Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video)+"_DM.csv", "w") as dm:
                    dm.write(dMap)