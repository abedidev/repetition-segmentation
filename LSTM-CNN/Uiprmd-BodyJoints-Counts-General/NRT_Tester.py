import datasetHandler
import VS_LSTM
import LSTM
import signal
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from os.path import exists
import math
from scipy import stats

exercise = int(input("Enter the exercise number: "))

with open("Exercises/"+str(exercise)+"/Settings.txt", 'r') as s:
    for line in s:
        line = line.split("\n")[0]
        data = line.split(" = ")[1]
        tag = line.split(" = ")[0]
        if tag == "num_layers":
            num_layers = int(data)
        elif tag == "batchSize":
            batchSize = int(data)

#Settings:
dataset = datasetHandler.LSTMdatasetWrapper("Exercises/"+str(exercise), "test_dataset.txt")
#total_samples = len(dataset)
#n_iterations = math.ceil(total_samples/batchSize)
inputSize = len(dataset[0][0][0])
model = VS_LSTM.LSTM(num_layers, inputSize*2, inputSize)
#model = LSTM.LSTM(1, len(dataset[0][0]), inputSize)
criterion = nn.MSELoss()
    
FILE = "Exercises/"+str(exercise)+"/checkpoint.pth"
checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])
epoch = checkpoint['epoch']
batchNum = checkpoint['batch_number']

print("\nTrained upto:")
print("epoch = "+str(epoch))
print("batchNum = "+str(batchNum))
print("batchSize = "+str(batchSize)+"\n")

train_loader = DataLoader(dataset=dataset,
                      batch_size=1,
                      shuffle=False,
                      num_workers=0)      

##########################################################
loss = 0
y_pred_List = []
labels_List = []
for i, (inputs, labels, seqLens) in enumerate(train_loader):
    seqLens = seqLens.view(seqLens.size(0))
    seqLens = [int(x) for x in seqLens]
    # Forward pass and loss
    y_pred = model(inputs, seqLens)
    
    #y_pred = model(inputs)
    #y_pred = y_pred.view(y_pred.size(0))
    
    labels = labels.view(labels.size(0))
    #labels = labels.long()
    
    loss += criterion(y_pred, labels)
    
    y_pred_List += [float(x) for x in y_pred]
    labels_List += [float(x) for x in labels]
SRCC = stats.spearmanr(y_pred_List, labels_List)[0]

print("y_pred : "+str(y_pred_List))
print("labels : "+str(labels_List))
print("Spearman's rank correlation coefficient: "+str(SRCC))
print("MSELoss : "+str(loss/len(dataset))+"\n")
##########################################################