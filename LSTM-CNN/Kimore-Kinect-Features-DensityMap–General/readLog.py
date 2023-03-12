import matplotlib.pyplot as plt
from os.path import exists
import os

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
        return [float(x) if isfloat(x) else x for x in st.split("[")[1].split("]")[0].split(", ")]
    
    sList = [x+("]"*factor) if x[len(x) - 1] != ']' else x for x in st[1:len(st)-1].split("]"*factor + ", ")]
    lst = []
    for s in sList:
        lst.append(strToList(s))
    return lst

exercise = int(input("Enter the exercise number: "))
y_pred = []
labels = []
with open("Exercises/"+str(exercise)+"/log.out", "r") as lg:
    for line in lg:
        line = line.split("\n")[0]
        if line[0:9] == "y_pred : ":
            y_pred = strToList(line[9:])
        if line[0:9] == "labels : ":
            labels = strToList(line[9:])

for fold in range(len(y_pred)):
    #Plotting y_pred[fold] and labels[fold]:
    fig, axs = plt.subplots(2)
    fig.subplots_adjust(hspace=0.8)
    fig.suptitle('Prediction Vs Ground Truth for Fold = '+str(fold))
    
    axs[0].set_title("Prediction for Fold = "+str(fold))
    axs[0].set_xlabel("Frame number")
    axs[0].set_ylabel("Predicted DM Values")
    axs[0].plot(list(range(len(y_pred[fold]))), y_pred[fold], color = 'r')
    
    
    axs[1].set_title("Ground truth for Fold = "+str(fold))
    axs[1].set_xlabel("Frame number")
    axs[1].set_ylabel("Ground truth DM values")
    axs[1].plot(list(range(len(labels[fold]))), labels[fold], color = 'b')
    
    if not exists("Exercises/"+str(exercise)+"/TestingPredVsGrdTruth"):
        os.system("mkdir Exercises/"+str(exercise)+"/TestingPredVsGrdTruth")
    plt.savefig("Exercises/"+str(exercise)+"/TestingPredVsGrdTruth"+"/Test_"+str(fold)+".png")
    plt.clf()
    plt.close(fig)