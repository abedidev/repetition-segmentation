import matplotlib.pyplot as plt
from os.path import exists

n=0
all_Losses = []
while(exists("Exercises/0/CV/log_"+str(n)+".out")):
    filename = "Exercises/0/CV/log_"+str(n)+".out"
    loss = []
    with open(filename, 'r') as f:
        for line in f:
            if line[0:3] == 'MSE':
                loss.append(float(line.split(',')[0][17:]))
    all_Losses.append(loss)
    n += 1
    
min_tests = min([len(loss) for loss in all_Losses])
avg_losses = []
for i in range(min_tests):
    avg_losses.append(0)
    for j in range(len(all_Losses)):
        avg_losses[i] += all_Losses[j][i]
    avg_losses[i] = avg_losses[i]/len(all_Losses)

x=[20*x for x in list(range(len(avg_losses)))]

plt.plot(x, avg_losses)
plt.xlabel('epoch')
plt.ylabel('Avg Testing MSE')
plt.title("Testing loss graph for 5-Fold Cross Validation")
plt.savefig("Exercises/0/CV/Avg_FFCV_MSE.png")