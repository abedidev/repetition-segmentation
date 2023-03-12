import matplotlib.pyplot as plt
import time

stop = 150
with open("../KiMoRe/CG/Expert/E_ID1/Es2/Raw/JointPosition011214_103845.csv", "r") as k:
    for j, line in enumerate(k):
        line = line.split("\n")[0]
        if line != '':
            x_list=[float(x) for i,x in enumerate(line.split(",")[:len(line.split(","))-1]) if i%4==0]
            y_list=[float(x) for i,x in enumerate(line.split(",")[:len(line.split(","))-1]) if i%4==1]
            plt.scatter(x_list, y_list, c='b')
            plt.title("E_ID1-Es2")
            plt.savefig("KinectPose.png")
            if j == stop:
                for i in range(len(x_list)):
                    point_x = x_list[i]
                    point_y = y_list[i]
                    plt.scatter([point_x], [point_y], c='r')
                    plt.annotate(str(i), (point_x, point_y))
                    plt.savefig("KinectPose.png")
            if j != stop:
                plt.clf()
            if j==stop:
                break