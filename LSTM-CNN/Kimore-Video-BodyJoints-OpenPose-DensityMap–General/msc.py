from os.path import exists
import os

exercise = 0
video = 1
dcr = 0
while(exists("Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video)+"_DM.csv")):
    j=1
    while(not exists("Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video+j)+"_DM.csv")):
        j += 1
        if j == 1000:
            exit()
    if(j != dcr+1):
        print(video)
        dcr += 1
    os.system("mv Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video+j)+"_DM.csv Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video+j-dcr)+"_DM.csv")
    video += 1