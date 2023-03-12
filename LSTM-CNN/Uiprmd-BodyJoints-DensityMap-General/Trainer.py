import datasetHandler
import VS_LSTM
#import LSTM
import signal
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from os.path import exists
import math
from scipy import stats

def signal_handler(sig, frame):
    checkpoint = {
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),
    "batch_number" : batchNum,
    }
    FILE = "Exercises/"+str(exercise)+"/checkpoint.pth"
    torch.save(checkpoint, FILE)
    
    with open("Exercises/"+str(exercise)+"/log.out", "a") as lg:
        lg.write('\nExiting...\n')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

exercise = int(sys.argv[1])

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:1'

with open("Exercises/"+str(exercise)+"/Settings.txt", 'r') as s:
    for line in s:
        line = line.split("\n")[0]
        data = line.split(" = ")[1]
        tag = line.split(" = ")[0]
        if tag == "learning_rate":
            learning_rate = float(data)
        elif tag == "num_epochs":
            num_epochs = int(data)
        elif tag == "num_layers":
            num_layers = int(data)
        elif tag == "batchSize":
            batchSize = int(data)
        elif tag == "printingBatch":
            printingBatch = int(data)

#Settings:
criterion = nn.MSELoss().to(device)
dataset = datasetHandler.LSTMdataset("Exercises/"+str(exercise), "train_dataset.txt")
dataset_test = datasetHandler.LSTMdataset("Exercises/"+str(exercise), "test_dataset.txt")
#total_samples = len(dataset)
#n_iterations = math.ceil(total_samples/batchSize)
inputSize = len(dataset[0][0][0])
model = VS_LSTM.LSTM(num_layers, inputSize*2, inputSize)
#model = LSTM.LSTM(1, len(dataset[0][0]), inputSize)
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
testingPeriod = 20 #Number of epochs after which the model is auto-tested

ch1 = str(sys.argv[2])

if torch.cuda.is_available():
    if ch1.upper() == "Y":
        with open("Exercises/"+str(exercise)+"/log.out", "w") as lg:
            lg.write("Device name: ")
            lg.write(str(torch.cuda.get_device_name(0))+"\n")
    else:
        with open("Exercises/"+str(exercise)+"/log.out", "a") as lg:
            lg.write("\nDevice name: ")
            lg.write(str(torch.cuda.get_device_name(0))+"\n")
else:
    if ch1.upper() == "Y":
        with open("Exercises/"+str(exercise)+"/log.out", "w") as lg:
            pass
        
epoch = 0
batchNum = 0
loss = "Dummy Initialisation"

#ch = input("Use existing datasets? Y/N: ")
ch = "Y"
if ch.upper() == "N":
    ds = datasetHandler.datasetHandler()
    ds.createDataset(exercise)
    ds.shuffleDataset("Exercises/"+str(exercise), "master_dataset.txt")
    ds.splitDataset("Exercises/"+str(exercise), "master_dataset.txt", exercise)
    
    with open("Exercises/"+str(exercise)+"/checkpoint.pth", "w") as c:
        pass
else:
    if ch1.upper() == "Y":
        with open("Exercises/"+str(exercise)+"/checkpoint.pth", "w") as c:
            pass
        with open("Exercises/"+str(exercise)+"/TrainingLoss.txt", "w") as tl:
            pass
    else:
        FILE = "Exercises/"+str(exercise)+"/checkpoint.pth"
        checkpoint = torch.load(FILE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        epoch = checkpoint['epoch']
        batchNum = checkpoint['batch_number']
        with open("Exercises/"+str(exercise)+"/checkpoint.pth", "w") as c:
            pass

with open("Exercises/"+str(exercise)+"/log.out", "a") as lg:
    lg.write("\nStarting from:\n")
    lg.write("epoch = "+str(epoch)+"\n")
    lg.write("batchNum = "+str(batchNum)+"\n")
    lg.write("batchSize = "+str(batchSize)+"\n\n")

train_loader = DataLoader(dataset=dataset,
                      batch_size=batchSize,
                      shuffle=False,
                      num_workers=0)      

while(epoch < num_epochs):    

    ##########################################################
    for i, (inputs, labels, seqLens) in enumerate(train_loader):
        if i == batchNum:
            seqLens = seqLens.view(seqLens.size(0))
            seqLens = [int(x) for x in seqLens]
            # Forward pass and loss
            y_pred = model(inputs, seqLens)
            
            #y_pred = model(inputs)
            #y_pred = y_pred.view(y_pred.size(0))
            
            labels = labels.view(labels.size(0))
            #labels = labels.long()
            
            loss = criterion(y_pred, labels)
            if batchNum == printingBatch:
                with open("Exercises/"+str(exercise)+"/TrainingLoss.txt", "a") as tl:
                    tl.write("Epoch : "+str(epoch)+"  BatchNum : "+str(i)+"  Loss : "+str(loss.item())+"\n")
                with open("Exercises/"+str(exercise)+"/log.out", "a") as lg:
                    lg.write("Epoch : "+str(epoch)+"  BatchNum : "+str(i)+"  Loss : "+str(loss.item())+"\n")
                    lg.write("\n")
                    lg.write("y_pred:\n")
                    lg.write(str(y_pred)+"\n")
                    lg.write("\n")
                    lg.write("labels:\n")
                    lg.write(str(labels)+"\n")
                    lg.write("\n\n")
            
            
            # Backward pass and update
            loss.backward()
            optimizer.step()  
                          
            # zero grad before new step
            optimizer.zero_grad()
            
            batchNum += 1

    ##########################################################
    if epoch%testingPeriod == 0:
        with open("Exercises/"+str(exercise)+"/log.out", "a") as lg:
            lg.write("Running test...\n")
            lg.write("Trained upto:\n")
            lg.write("epoch = "+str(epoch)+"\n")
            lg.write("batchNum = "+str(batchNum - 1)+"\n")
            lg.write("batchSize = "+str(batchSize)+"\n\n")

        train_loader_test = DataLoader(dataset=dataset_test,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)      

        ##########################################################
        loss = 0
        y_pred_List = []
        labels_List = []
        for i, (inputs, labels, seqLens) in enumerate(train_loader_test):
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

        with open("Exercises/"+str(exercise)+"/log.out", "a") as lg:
            lg.write("y_pred : "+str(y_pred_List)+"\n")
            lg.write("labels : "+str(labels_List)+"\n")
            lg.write("Spearman's rank correlation coefficient: "+str(SRCC)+"\n")
            lg.write("MSELoss : "+str(loss/len(dataset_test))+"\n\n\n")
        ##########################################################
    batchNum = 0
    epoch += 1

signal_handler(0, 0)