import os
from os.path import exists

for video in range(1, 347):
    with open("Exercises/0/videos/V"+str(video)+"/Features.txt", "r") as f:
        num_frames = 0
        for line in f:
            num_frames += 1
        dMap = ""
        with open("Annotations/Exercise_0/DensityMaps/V"+str(video)+"_DM.csv", "r") as dm:
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
        with open("Annotations/Exercise_0/DensityMaps/V"+str(video)+"_DM.csv", "w") as dm:
            dm.write(dMap)