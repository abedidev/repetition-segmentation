from os.path import exists
import matplotlib.pyplot as plt
import csv

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
    

fold = 0
header = ["Video Index", 'Actual scores', 'Predicted scores']
y_pred = []
labels = []
y_pred_temp=[]
labels_temp=[]
while exists("Exercises/0/CV/log_"+str(fold)+".out"):
    filename = "Exercises/0/CV/log_"+str(fold)+".out"
    with open(filename, 'r') as f:
        for line in f:
            if line[0:10] == "y_pred : [":
                y_pred_temp = strToList(line[9:].split("\n")[0])
            if line[0:10] == "labels : [":
                labels_temp = strToList(line[9:].split("\n")[0])
        y_pred += y_pred_temp
        labels += labels_temp
    fold += 1

plt.figure(figsize=(70, 7))
plt.rcParams.update({'font.size': 17})
plt.scatter(list(range(len(labels))), labels, color = "b")
plt.scatter(list(range(len(y_pred))), y_pred, color = "r")
plt.legend(['Actual RepCounts', "Predicted RepCounts"])
plt.plot(list(range(len(labels))), labels)
plt.plot(list(range(len(y_pred))), y_pred)
plt.xlabel("Video Index")
plt.ylabel("RepCounts")
plt.savefig("Exercises/0/CV/PredVsTruth.png")
plt.clf()

data = []
for i in range(len(labels)):
    lst = [i, labels[i], y_pred[i]]
    data.append(lst)
with open("Exercises/0/CV/PredVsTruth.csv", 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
