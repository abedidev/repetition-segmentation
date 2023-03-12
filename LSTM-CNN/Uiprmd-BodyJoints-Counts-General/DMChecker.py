from os.path import exists
import os

ex = 0
while(exists("Annotations/Exercise_"+str(ex)+"/DensityMaps")):
    for video in range(1,201):
        with open("Annotations/Exercise_"+str(ex)+"/DensityMaps/V"+str(video)+"_DM.csv", "r") as dm:
            dmap = dm.readline()
            dmap = [float(x) for x in dmap.split("\n")[0].split(",")]
            if(round(sum(dmap))<10):
                print("gaya")
    ex += 1