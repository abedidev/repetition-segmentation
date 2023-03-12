import time
import signal
import sys
from os.path import exists
import os
import torch
#import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random
import shutil
#import openpyxl
from pathlib import Path

def getVideoNumber(personID, exercise):
    video = 1
    while(exists("Exercises/"+str(exercise)+"/videos/V"+str(video))):
        if exists("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/"+str(personID)+".xlsx"):
            return video
        video += 1
    return -1

def signal_handler(sig, frame):
    print('\nExiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class datasetHandler():
    
    def isfloat(self, num):
        try:
            float(num)
            return True
        except ValueError:
            return False
    
    def strToList(self, st): #WARNING: THIS FUNCTION IS DIFFERENT FROM THE OTHER strToList functions...
        if st == '[]':
            return []
        factor = -1
        for ch in st:
            if ch != '[':
                break
            factor += 1
        if factor == 0:
            return [float(x) if self.isfloat(x) else x for x in st.split("[")[1].split("]")[0].split(", ")]
        
        sList = [x+("]"*factor) if x[len(x) - 1] != ']' else x for x in st[1:len(st)-1].split("]"*factor + ", ")]
        lst = []
        for s in sList:
            lst.append(self.strToList(s))
        return lst
    
    def readExerciseTS(self, exercise, video):
        files = os.listdir("Exercises/"+str(exercise)+"/videos/V"+str(video))
        filename = ""
        for file in files:
            if file[-5:len(file)] == '.xlsx':
                filename = file
                break
        path = "Exercises/"+str(exercise)+"/videos/V"+str(video)+"/"+filename
        cell = {
            0 : "B2",
            1 : "C2",
            2 : "D2",
            3 : "E2",
            4 : "F2"
        }
        xlsx_file = Path(path)
        wb_obj = openpyxl.load_workbook(xlsx_file) 

        # Read the active sheet:
        sheet = wb_obj.active
        return [sheet[cell[exercise]].value, filename[0:-5]]
    
    def createDataset(self, exercise):
        with open("Exercises/"+str(exercise)+"/NO_DMAPS.txt", "w") as ns:
            pass
        with open("Exercises/"+str(exercise)+"/master_dataset.txt", "w") as md:
            pass
        
        video = 1
        while(exists("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/Features.txt")):
            self.appendToDataset(exercise, video)
            video += 1
    
    def appendToDataset(self, exercise, video):
        _, person = self.readExerciseTS(exercise, video)
        DMap = -1
        if exists("Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video)+"_DM.csv"):
            with open("Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video)+"_DM.csv", "r") as dm:
                DMap = [float(x) if self.isfloat(x) else "" for x in dm.readline().split("\n")[0].split(",")]
        if DMap == -1:
            DMap = None
        if DMap == None:
            with open("Exercises/"+str(exercise)+"/NO_DMAPS.txt", "a") as ns:
                with open("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/Features.txt", "r") as f:
                    frameSequence = []
                    for line in f:
                        features = []
                        line = line.split("\n")[0]
                        line = line.replace("'None'", '"None"')
                        line = [line.split("['")[x].split("', ")[1].split("], ")[0].split("]]")[0] if x!=0 else x for x in range(len(line.split("['")))][1:]
                        line[len(line)-1]+=']'
                        line = [self.strToList(x) for x in line]
                        for feature in line:
                            features += feature
                        features = [x if x!='"None"' else 0 for x in features]
                        frameSequence.append(features)
                    if len(frameSequence) != 0:
                        ns.write("V"+str(video)+":"+person+":"+str(len(frameSequence))+":"+str(frameSequence)+"\n")
        else:
            with open("Exercises/"+str(exercise)+"/master_dataset.txt", "a") as md:
                with open("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/Features.txt", "r") as f:
                    frameSequence = []
                    for line in f:
                        features = []
                        line = line.split("\n")[0]
                        line = line.replace("'None'", '"None"')
                        line = [line.split("['")[x].split("', ")[1].split("], ")[0].split("]]")[0] if x!=0 else x for x in range(len(line.split("['")))][1:]
                        line[len(line)-1]+=']'
                        line = [self.strToList(x) for x in line]
                        for feature in line:
                            features += feature
                        features = [x if x!='"None"' else 0 for x in features]
                        frameSequence.append(features)
                    if len(frameSequence) != 0:
                        md.write("V"+str(video)+":"+person+":"+str(len(frameSequence))+":"+str(DMap)+":"+str(frameSequence)+"\n")
        
    def shuffleDataset(self, folder, filename): #Shuffle the lines in datasets/master_dataset.txt
                              #We can't store all the contents of datasets/master_dataset.txt in some list due to memory constraint
        
        lines = open(folder+"/"+filename).readlines()
        random.shuffle(lines)
        open(folder+"/"+filename,'w').writelines(lines)
    
    def splitDataset(self, folder, filename, exercise, test_videos):
        with open(folder+"/"+filename, "r") as md:
            with open(folder+"/train_dataset.txt", "w") as mtrain:
                with open(folder+"/test_dataset.txt", "w") as mtest:
                    for line in md:
                        pID = line.split(":")[0]
                        if "V"+str(getVideoNumber(pID, exercise)) in test_videos:
                            mtest.write(line)
                        else:
                            mtrain.write(line)
        
        
class LSTMdataset(Dataset):
    
    def __init__(self, folder, filename):
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:1'
        xy = np.loadtxt(folder+"/"+filename, delimiter=":", dtype = str)
        if len(xy.shape) == 1:
            xy = xy.reshape((1,xy.shape[0]))
        self.n_samples = xy.shape[0]
        seqList = []
        for seq in xy[:, 4:]:
            seqList.append(self.strToList(seq[0]))
            
        dMapList = []
        for dMap in xy[:, 3]:
            dMapList.append(self.strToList(dMap))
        dMapList = self.padDMapList(dMapList)

        # here the first column is the class label, the rest is the frame sequence
        #self.x_data = torch.tensor(seqList, dtype=torch.float32) # size [n_samples, n_time_steps, n_features]
        self.x_data = self.padData(seqList)
        self.y_data = torch.tensor(dMapList, dtype = torch.float32).to(self.device)
        self.l_data = torch.from_numpy(xy[:, [2]].astype(np.float32)).to(self.device)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.l_data[index]

    # call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    def strToList(self, st):
        if st == '[]':
            return []
        factor = -1
        for ch in st:
            if ch != '[':
                break
            factor += 1
        if factor == 0:
            return [float(x) for x in st.split("[")[1].split("]")[0].split(", ")]
        
        sList = [x+("]"*factor) if x[len(x) - 1] != ']' else x for x in st[1:len(st)-1].split("]"*factor + ", ")]
        lst = []
        for s in sList:
            lst.append(self.strToList(s))
        return lst
    
    def padData(self, X_list):
        max_len = 0
        num_features = 0
        for seq in X_list:
            if len(seq) != 0:
                num_features = len(seq[0])
            if len(seq) > max_len:
                max_len = len(seq)

        padList = [0]*num_features

        for i in range(len(X_list)):
            iter = max_len - len(X_list[i])
            for j in range(iter):
                X_list[i].append(padList)

        X = torch.tensor(X_list, dtype = torch.float32).to(self.device)

        #print(X)
        return X
    
    def padDMapList(self, dMapList):
        max_len = 0
        for dMap in dMapList:
            if len(dMap) > max_len:
                max_len = len(dMap)
        for i in range(len(dMapList)):
            dMapList[i] = dMapList[i] + [0]*(max_len - len(dMapList[i]))
        return dMapList

class LSTMdatasetWrapper(Dataset):
    def __init__(self, folder, filename):
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:1'
        
        count = 0
        with open(folder+"/"+filename, "r") as od:
            for line in od:
                with open(folder+"/"+"dummyDataset_"+str(count)+".txt", "w") as dd:
                    dd.write(line)
                    count += 1
        self.datasets = []
        for i in range(count):
            self.datasets.append(LSTMdataset(folder, "dummyDataset_"+str(i)+".txt"))
            os.remove(folder+"/"+"dummyDataset_"+str(i)+".txt")
        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.datasets[index][0]

    # call len(dataset) to return the size
    def __len__(self):
        return len(self.datasets)
            
        
def main():
    ds = datasetHandler()
    exercise = int(input("Enter the exercise number: "))
    ds.createDataset(exercise)
    ds.shuffleDataset("Exercises/"+str(exercise), "master_dataset.txt")
    #ds.splitDataset("Exercises/"+str(exercise), "master_dataset.txt", exercise)
    
if __name__ == "__main__":
    main()
