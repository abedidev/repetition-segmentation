from os.path import exists
from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt

def strToList(st):
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
        lst.append(strToList(s))
    return lst

def findPeaks(data): #Returns maxima indices
    #apply a Savitzky-Golay filter
    smooth = savgol_filter(data, window_length = 29, polyorder = 4)

    #find the maximums
    peaks_idx_max, _ = find_peaks(smooth, prominence = 0.001)
    
    return peaks_idx_max

def getReport(predCounts, grdthCounts):
    #print(grdthCounts)
    mae = 0
    mse = 0
    obo = 0
    acc = 0
    for i in range(len(predCounts)):
        mae = mae + abs(predCounts[i] - grdthCounts[i])
        mse = mse + (abs(predCounts[i] - grdthCounts[i]) * abs(predCounts[i] - grdthCounts[i]))
        if(abs(predCounts[i] - grdthCounts[i]) <= 1):
            obo += 1
        if(predCounts[i] == grdthCounts[i]):
            acc += 1
    mae = mae/len(predCounts)
    mse = mse/len(predCounts)
    obo = obo/len(predCounts)
    acc = acc/len(predCounts)
    print("MAE: "+str(mae))
    print("MSE: "+str(mse))
    print("Accuracy: "+str(acc))
    print("OBO: "+str(obo))

data_dir = input("Enter the data_dir: ")
model_type = int(input("Enter 1 for density map model, 0 for counts model: "))
epochs = int(input("Enter the number of epochs at which results are needed (-100 for last epoch): "))
epochs = int(epochs/20)*20 #Greatest multiple of 20 less than or equal to entered num_epochs

y_pred = []
labels = []

log = 0
while(exists(data_dir+"log_"+str(log)+".out")):
    with open(data_dir+"log_"+str(log)+".out", "r") as lg:
        y_pred_temp = []
        labels_temp = []
        for line in lg:
            line = line.split("\n")[0]
            if line[:8] == "y_pred :":
                y_pred_temp = strToList(line[9:])
            if line[:8] == "labels :":
                labels_temp = strToList(line[9:])
            if line[:7] == "epoch =" and int(line[8:]) == epochs+20:
                break
        y_pred = y_pred + y_pred_temp
        labels = labels + labels_temp
    log += 1

if model_type == 1:
    predicted_counts = [len(findPeaks(x)) for x in y_pred]
    grdth_counts = [round(sum(x)) for x in labels]
    evaluator_grdth_counts = [len(findPeaks(x)) for x in labels]

    print("\n")
    print("Test results at epoch="+str(epochs)+": ")
    getReport(predicted_counts, grdth_counts)
    print("\n")
    print("Evaluator quality check: ")
    getReport(evaluator_grdth_counts, grdth_counts)
    print("\n")

elif model_type == 0:
    predicted_counts = [round(x) for x in y_pred]
    grdth_counts = [round(x) for x in labels]

    print("\n")
    print("Test results at epoch="+str(epochs)+": ")
    getReport(predicted_counts, grdth_counts)
    print("\n")
    
else:
    print("Wrong model type!!")