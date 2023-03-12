import pickle

f=open("dm_feats_v1.pkl", "rb")
data = pickle.load(f)

for i in range(1):
   for key in data[i][1].keys():
       value = data[i][1][key]
       if(sum(value)<=9):
           print("Gaya")

# value = data[0][1].keys()
# print(value)