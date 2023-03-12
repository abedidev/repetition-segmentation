import matplotlib.pyplot as plt
from os.path import exists
dMapNum = 1
while(exists("V"+str(dMapNum)+"_DM.csv")):
    dMap = []
    with open("V"+str(dMapNum)+"_DM.csv", 'r') as dm:
        dMap = [float(x) for x in dm.readline().split("\n")[0].split(",")]

    x = list(range(len(dMap)))
    y = dMap
    plt.plot(x,y)
    plt.xlabel("Frame Number")
    plt.ylabel("Density Map value")
    plt.savefig("V"+str(dMapNum)+"_DM_plot.png")
    plt.clf()
    dMapNum += 1