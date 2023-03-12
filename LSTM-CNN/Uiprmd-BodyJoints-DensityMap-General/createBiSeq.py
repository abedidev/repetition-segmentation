#NOTE: THIS SCRIPT IS TO BE EXECUTED AFTER FEATURES ARE EXTRACTED FOR EACH VIDEO OF THE EXERCISE

from os.path import exists
import os
import cv2
exercise = input("Enter the exercise number: ")

if not exists("Annotations/Exercise_"+str(exercise)+"/DensityMaps"):
    os.system("mkdir "+"Annotations/Exercise_"+str(exercise)+"/DensityMaps")
    
video = 1
while(exists("Annotations/Exercise_"+str(exercise)+"/csv_files/V"+str(video)+".csv")):
    o_fps = 5 #Dummy initialisation
    cap = cv2.VideoCapture("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/V"+str(video)+".mp4")
        
    # Finding OpenCV version:
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        o_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS) 
    else :
        o_fps = cap.get(cv2.CAP_PROP_FPS)
        
    #num_frames = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
    #Using Features.txt file of the video to determine num_frames:
    num_frames = 0
    with open("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/Features.txt", "r") as f:
        num_frames = len(f.readlines())
    
    bi_seq = ""
    start_frame_num = 0
    end_frame_num = 0 
    tot_frame_count = -1
       
    with open("Annotations/Exercise_"+str(exercise)+"/csv_files/V"+str(video)+".csv", "r") as cs:
        cs.readline()
        line_count = 0
        for line in cs:
            line = line.split("\n")[0]
            
            start_time = line.split(",")[0]
            end_time = line.split(",")[1]
            
            start_time_sec = sum([float(x)*pow(60,2-i) for i, x in enumerate(start_time.split(":"))])
            end_time_sec = sum([float(x)*pow(60,2-i) for i, x in enumerate(end_time.split(":"))])
            
            start_frame_num = int(o_fps * start_time_sec)
            end_frame_num = int(o_fps * end_time_sec)
            
            if line_count != 0:
                start_frame_num += 1
            
            bi_seq = bi_seq + ",0"*(start_frame_num - tot_frame_count - 1)
            bi_seq += ",1"+",0"*(end_frame_num-start_frame_num-1)+",1"
            tot_frame_count = end_frame_num
            
            line_count += 1
    bi_seq = bi_seq + ",0"*(num_frames - end_frame_num - 1) #appending 0's to the end of bi_seq
    bi_seq = bi_seq[1:]
    
    if num_frames != len(bi_seq.split(",")):
        lst = bi_seq.replace(",", "").split("1")
        lst = [len(x) for x in lst]
        frac = (num_frames - len(bi_seq.split(",")))/sum(lst)
        lst = [int((1+frac)*x) for x in lst]
        
        factor = int((num_frames - (sum(lst)+bi_seq.count('1')))/abs(num_frames - (sum(lst)+bi_seq.count('1'))))
        i = 0
        while(sum(lst)+bi_seq.count('1') != num_frames):
            print("sum(lst)+bi_seq.count('1') : "+str(sum(lst)+bi_seq.count('1')), end = "  Vs  ")
            print("num_frames : "+str(num_frames))
            print("factor : "+str(factor)+"\n")
            lst[i] = max(lst[i] + factor, 0)
            i = (i + 1)%len(lst)
            
        lst = ['0'*x for x in lst]
        bi_seq = "1".join(lst)
        bi_seq = list(bi_seq)
        bi_seq = ",".join(bi_seq)
        
    if num_frames != len(bi_seq.split(",")):
        print("Num_frames: "+str(num_frames))
        print("BiSeq length: "+str(len(bi_seq.split(","))))
        print("BiSeq: "+bi_seq)
        print("ERROR for video="+str(video)+" : Total number of frames not equal to binary sequence length!")
        break
    with open("Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video)+"_DM.csv", "w") as dm:
        dm.write(bi_seq+"\n")
    video += 1