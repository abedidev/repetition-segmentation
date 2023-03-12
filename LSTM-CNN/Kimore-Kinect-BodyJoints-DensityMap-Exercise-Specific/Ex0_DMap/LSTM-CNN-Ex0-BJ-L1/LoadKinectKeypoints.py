from os.path import exists
import os

exercise = 0
while(exists("Exercises/"+str(exercise))):
    video = 1
    while(exists("Exercises/"+str(exercise)+"/videos/V"+str(video))):
        with open("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/Keypoints.txt", "w") as kp:
            files = os.listdir("Exercises/"+str(exercise)+"/videos/V"+str(video))
            personID = ""
            for file in files:
                if file[len(file)-5:] == ".xlsx":
                    personID = file[:len(file)-5]
                    break
            for dir_l1 in set(set(os.listdir("KiMoRe")) - set(['.DS_Store'])):
                for dir_l2 in set(set(os.listdir("KiMoRe/"+dir_l1)) - set(['.DS_Store'])):
                    for dir_l3 in set(set(os.listdir("KiMoRe/"+dir_l1+"/"+dir_l2)) - set(['.DS_Store'])):
                        if dir_l3 == personID:
                            if exists("KiMoRe/"+dir_l1+"/"+dir_l2+"/"+dir_l3+"/Es"+str(exercise+1)):
                                files = os.listdir("KiMoRe/"+dir_l1+"/"+dir_l2+"/"+dir_l3+"/Es"+str(exercise+1)+"/Raw")
                                kpFile = ""
                                for file in files:
                                    if file[:13] == 'JointPosition':
                                        kpFile = file
                                        break
                                if kpFile == "":
                                    print("No Kinect Joint Position file in dir="+"KiMoRe/"+dir_l1+"/"+dir_l2+"/"+dir_l3+"/Es"+str(exercise+1)+"/Raw")
                                    continue
                                with open("KiMoRe/"+dir_l1+"/"+dir_l2+"/"+dir_l3+"/Es"+str(exercise+1)+"/Raw/"+kpFile, "r") as kinectJP:
                                    for line in kinectJP:
                                        line = line.split("\n")[0]
                                        if line != '':
                                            keypoint = []
                                            keypointList = []
                                            for i, val in enumerate(line[:len(line)-1].split(",")):
                                                if i%4 == 0:
                                                    keypoint = []
                                                    keypoint.append(int(i/4))
                                                    keypoint.append(float(val))
                                                elif i%4 == 1:
                                                    keypoint.append(float(val))
                                                elif i%4 == 2:
                                                    keypoint.append(float(val))
                                                else:
                                                    keypoint.append(1.0) #Assuming visibility/confidence = 1 for all kinect keypoints
                                                    keypointList.append(keypoint)
                                            kp.write(str(keypointList)+"\n")
        video += 1
    exercise += 1
