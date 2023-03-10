import numpy as np
import torch

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

def padData(X_list):
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
    
    X = torch.tensor(X_list, dtype = torch.float32)
    
    #print(X)
    return X

exercise = int(input("Enter the exercise number (valid values: 0,1,2,3,4): "))
xy = np.loadtxt("Exercises/"+str(exercise)+"/master_dataset.txt", delimiter=":", dtype = str)
n_samples = xy.shape[0]
seqList = []
for seq in xy[:, 3:]:
    seqList.append(strToList(seq[0]))
                   
x_data = seqList #List of feature sequences
padded_x_data = padData(x_data) #Tensor of feature sequences after zero padding
y_data = torch.from_numpy(xy[:, [2]].astype(np.float32)) #Tensor of corresponding total score values
l_data = torch.from_numpy(xy[:, [1]].astype(np.float32)) #Tensor of corresponding feature sequence lengths before padding
p_data = xy[:, [0]] #Numpy array of corresponding person name/ID values

print("Tensor containing padded frame sequences:")
print(padded_x_data)
torch.save(padded_x_data, "Exercises/"+str(exercise)+"/padded_sequences.pt")
print("\nTensor containing the corresponding total score values:")
print(y_data)
torch.save(y_data, "Exercises/"+str(exercise)+"/scores.pt")
print("\nTensor containing the corresponding feature sequence lengths before padding:")
print(l_data)
torch.save(l_data, "Exercises/"+str(exercise)+"/sequence_lengths.pt")
print("\nNumpy array containing the corresponding person name/ID values:")
print(p_data)
np.save("Exercises/"+str(exercise)+"/person_IDs.npy", p_data)