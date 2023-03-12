import pickle
from os.path import exists
import os

with open("dm_feats_v1.pkl", 'rb') as dm:
    data = pickle.load(dm)
    for ex, dMaps in data:
        exercise = ex.split("_")[1]
        if not exists("Annotations/Exercise_"+str(exercise)):
            os.system("mkdir "+"Annotations/Exercise_"+str(exercise))
        if not exists("Annotations/Exercise_"+str(exercise)+"/DensityMaps"):
            os.system("mkdir "+"Annotations/Exercise_"+str(exercise)+"/DensityMaps")
        for key in dMaps.keys():
            video = key.split(".")[0][1:]
            with open("Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video)+"_DM.csv", 'w') as vdm:
                with open("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/Features.txt", 'r') as f:
                    num_frames = len(f.readlines())
                    dMap = [str(x) for x in dMaps[key]]
                    if len(dMap) > num_frames:
                        dMap = dMap[len(dMap)-num_frames:len(dMap)]
                    elif len(dMap) < num_frames:
                        for i in range(num_frames-len(dMap)):
                            dMap.append('0')
                    dMap = ",".join(dMap)
                    vdm.write(dMap+"\n")