import os
import sys
from os.path import exists
import time
import signal

def signal_handler(sig, frame):
    print("Exiting...")
    sys.exit(0)

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
    
def strToList(st): #WARNING: THIS FUNCTION IS DIFFERENT FROM THE OTHER strToList functions...
    if st == '[]':
        return []
    factor = -1
    for ch in st:
        if ch != '[':
            break
        factor += 1
    if factor == 0:
        return [int(x) if isfloat(x) else x for x in st.split("[")[1].split("]")[0].split(", ")]

    sList = [x+("]"*factor) if x[len(x) - 1] != ']' else x for x in st[1:len(st)-1].split("]"*factor + ", ")]
    lst = []
    for s in sList:
        lst.append(strToList(s))
    return lst

def getTrainingList():
    stateList = []
    with open("Exercises/"+str(exercise)+"/CV/FoldState.txt", 'r') as fs:
        stateList = strToList(fs.readline().split("\n")[0])
    trainingList = []
    for fold in stateList:
        if fold[1] == foldState["training"]:
            trainingList.append(fold)
    return trainingList

def getPausedList():
    stateList = []
    with open("Exercises/"+str(exercise)+"/CV/FoldState.txt", 'r') as fs:
        stateList = strToList(fs.readline().split("\n")[0])
    pausedList = []
    for fold in stateList:
        if fold[1] == foldState["paused"]:
            pausedList.append(fold)
    return pausedList

signal.signal(signal.SIGINT, signal_handler)

exercise = int(sys.argv[1])
CVFolds = 5
MAX_PARALLEL_EXECUTION = 3 #MAX_PARALLEL_EXECUTION = 9
TRAINING_TIME_SLOT_MINS = 15 #TRAINING_TIME_SLOT_MINS = 60
dummyPID = -1
foldState = {
    "training" : 0,
    "paused" : 1,
    "finished" : 2
}

if (not exists("Exercises/"+str(exercise)+"/CV")):
    os.mkdir("Exercises/"+str(exercise)+"/CV")
    
if not exists("Exercises/"+str(exercise)+"/CV/FoldState.txt"):
    with open("Exercises/"+str(exercise)+"/CV/FoldState.txt", 'w') as fs:
        stateList = []
        for i in range(CVFolds):
            state = [i, foldState["paused"], dummyPID]
            stateList.append(state)
        fs.write(str(stateList)+"\n")
        
training = getTrainingList()
for fold in training:
    os.kill(fold[2], signal.SIGINT)
    time.sleep(60)
    
start_time = time.time()
end_time = start_time + (60*TRAINING_TIME_SLOT_MINS)
nextTrainingFoldNo = -1
while(True):
    if (end_time - start_time)/60 >= TRAINING_TIME_SLOT_MINS:
        start_time = end_time
        training = getTrainingList()
        paused = getPausedList()
        if len(paused) == 0:
            break
        if len(training) == 0:
            for fold in range(min(MAX_PARALLEL_EXECUTION, len(paused))):
                if exists("Exercises/"+str(exercise)+"/CV/checkpoint_"+str(paused[fold][0])+".pth"):
                    trainingFromScratch = 'n'
                else:
                    trainingFromScratch = 'y'
                os.system("nohup python CV_Trainer.py "+str(exercise)+" "+trainingFromScratch+" "+str(paused[fold][0])+" &")
                time.sleep(60)
            if MAX_PARALLEL_EXECUTION < len(paused):
                nextTrainingFoldNo = paused[MAX_PARALLEL_EXECUTION][0]
            else:
                nextTrainingFoldNo = -1
        else:
            for fold in training:
                os.kill(fold[2], signal.SIGINT)
                time.sleep(60)
            training = getTrainingList()
            paused = getPausedList()
            for index, fold in enumerate(paused):
                if fold[0] == nextTrainingFoldNo:
                    break
            for fold in range(min(MAX_PARALLEL_EXECUTION, len(paused))):
                i = (index + fold) % len(paused)
                if exists("Exercises/"+str(exercise)+"/CV/checkpoint_"+str(paused[i][0])+".pth"):
                    trainingFromScratch = 'n'
                else:
                    trainingFromScratch = 'y'
                os.system("nohup python CV_Trainer.py "+str(exercise)+" "+trainingFromScratch+" "+str(paused[i][0])+" &")
                time.sleep(60)
            if MAX_PARALLEL_EXECUTION < len(paused):
                i = (index + MAX_PARALLEL_EXECUTION) % len(paused)
                nextTrainingFoldNo = paused[i][0]
            else:
                nextTrainingFoldNo = -1
        if nextTrainingFoldNo == -1:
            break
    
    end_time = time.time()
            
        
