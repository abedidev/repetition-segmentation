from os.path import exists
import os

for i in range(1,5):
    video = 1
    while(exists("Annotations/Exercise_"+str(i)+"/DensityMaps/V"+str(video)+"_DM.csv")):
        num = len(os.listdir("Annotations/Exercise_0/DensityMaps")) + 1
        os.system("mv Annotations/Exercise_"+str(i)+"/DensityMaps/V"+str(video)+"_DM.csv Annotations/Exercise_0/DensityMaps/V"+str(num)+"_DM.csv")
        video += 1
        
#for i in range(1,5):
#    video = 1
#    while(exists("Exercises/"+str(i)+"/videos/V"+str(video))):
#        num = len(os.listdir("Exercises/0/videos")) + 1
#        os.system("mv Exercises/"+str(i)+"/videos/V"+str(video)+" Exercises/0/videos/V"+str(num))
#        video += 1
#
#v=74
#while(exists("Exercises/0/videos/V"+str(v))):
#    os.system("mv Exercises/0/videos/V"+str(v)+" Exercises/0/videos/V"+str(v-1))
#    v += 1