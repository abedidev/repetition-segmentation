from fileinput import filename
import cv2 
import mediapipe as mp
import time
import signal
import sys
from os.path import exists
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random
import shutil
import openpyxl
from pathlib import Path

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
        with open("Exercises/"+str(exercise)+"/NO_SCORES.txt", "w") as ns:
            pass
        with open("Exercises/"+str(exercise)+"/master_dataset.txt", "w") as md:
            pass
        
        video = 1
        while(exists("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/Features.txt")):
            self.appendToDataset(exercise, video)
            video += 1
    
    def appendToDataset(self, exercise, video):
        score, person = self.readExerciseTS(exercise, video)
        if score == None:
            with open("Exercises/"+str(exercise)+"/NO_SCORES.txt", "a") as ns:
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
                        ns.write(person+":"+str(len(frameSequence))+":"+str(frameSequence)+"\n")
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
                        md.write(person+":"+str(len(frameSequence))+":"+str(score)+":"+str(frameSequence)+"\n")
        
    def shuffleDataset(self, folder, filename): #Shuffle the lines in datasets/master_dataset.txt
                              #We can't store all the contents of datasets/master_dataset.txt in some list due to memory constraint
        
        lines = open(folder+"/"+filename).readlines()
        random.shuffle(lines)
        open(folder+"/"+filename,'w').writelines(lines)
    
    def splitDataset(self, folder, filename):
        #split datasets/master_dataset.txt into datasets/dataset_test.txt, datasets/dataset_1.txt, datasets/dataset_2.txt, ... datasets/dataset_n.txt
        #datasets/dataset_test.txt must contain exactly x lines, where x = (number of lines in datasets/master_dataset.txt)/13
        #datasets/dataset_1.txt, datasets/dataset_2.txt, ... datasets/dataset_n.txt each must contain 720 lines at max
        #Open datasets/dataset_1.txt, datasets/dataset_2.txt, ... datasets/dataset_n.txt files in "w" mode instead of "a" mode
        #The value of n = ceil((number of lines in datasets/master_dataset.txt - x) / 720)
        
        with open(folder+"/"+filename, 'r') as f:
            total_no_of_lines = len(f.readlines())
        
        tr = math.ceil(total_no_of_lines*0.9)
        
        with open(folder+"/"+filename, 'r') as m:
            with open(folder+"/"+"train_dataset.txt", 'w') as fTr:
                with open(folder+"/"+"test_dataset.txt", 'w') as fTe:
                    for line in m:
                        if tr > 0:
                            fTr.write(line)
                            tr -= 1
                        else:
                            fTe.write(line)
        
        
class LSTMdataset(Dataset):
    
    def __init__(self, folder, filename):
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        xy = np.loadtxt(folder+"/"+filename, delimiter=":", dtype = str)
        if len(xy.shape) == 1:
            xy = xy.reshape((1,xy.shape[0]))
        self.n_samples = xy.shape[0]
        seqList = []
        for seq in xy[:, 3:]:
            seqList.append(self.strToList(seq[0]))

        # here the first column is the class label, the rest is the frame sequence
        #self.x_data = torch.tensor(seqList, dtype=torch.float32) # size [n_samples, n_time_steps, n_features]
        self.x_data = self.padData(seqList)
        self.y_data = torch.from_numpy(xy[:, [2]].astype(np.float32)).to(self.device) # size [n_samples, 1]
        self.l_data = torch.from_numpy(xy[:, [1]].astype(np.float32)).to(self.device)

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
        
def main():
    ds = datasetHandler()
    exercise = int(input("Enter the exercise number: "))
    ds.createDataset(exercise)
    ds.shuffleDataset("Exercises/"+str(exercise), "master_dataset.txt")
    ds.splitDataset("Exercises/"+str(exercise), "master_dataset.txt")
    
if __name__ == "__main__":
    main()