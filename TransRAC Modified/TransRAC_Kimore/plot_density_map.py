import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)
plt.rcParams["figure.figsize"] = 5,2

def plotDensityMap(y, name):

    x = 1 + np.arange(len(y))

    fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
    ax.imshow(y[np.newaxis,:], cmap="Reds", aspect="auto", extent=extent)
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])

    ax2.plot(x,y)

    plt.tight_layout()
    plt.savefig(name + '.png')

#if __name__ == '__main__':
#    y = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0015, 0.0039, 0.0089, 0.0182, 0.0334,
#         0.0549, 0.0807, 0.1062, 0.1253, 0.1324, 0.1253, 0.1062, 0.0807, 0.0549,
#         0.0334, 0.0182, 0.0089, 0.0039, 0.0019, 0.0056, 0.0146, 0.0320, 0.0603,
#         0.0968, 0.1327, 0.1554, 0.1554, 0.1327, 0.0968, 0.0603, 0.0320, 0.0146,
#         0.0056, 0.0019, 0.0056, 0.0146, 0.0320, 0.0603, 0.0968, 0.1327, 0.1554,
#         0.1554, 0.1327, 0.0968, 0.0603, 0.0320, 0.0146, 0.0056, 0.0080, 0.1069,
#         0.3849, 0.3849, 0.1069, 0.0080, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#         0.0000])
#    print(sum(y))
#    plotDensityMap(y, 'test')
