import os
from os.path import exists

video = 1
while(exists("Exercises/0/videos/V"+str(video))):
    files = os.listdir("Exercises/0/videos/V"+str(video))
    for file in files:
        if file[-3:] == "mp4":
            if file.split(".")[0].split("V")[1] != str(video):
                os.system("mv Exercises/0/videos/V"+str(video)+"/"+file+" Exercises/0/videos/V"+str(video)+"/V"+str(video)+".mp4")
    video += 1