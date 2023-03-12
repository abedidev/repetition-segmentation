import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm

np.random.seed(1)
plt.rcParams["figure.figsize"] = 5, 2

def plotDMSequence(y_pred, y_real, folder, name, video_name):
    x = 1 + np.arange(len(y_pred))

    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(y_pred, color = 'r')
    axs[0].set_title('Predicted Density map')
    axs[1].plot(y_real, color = 'b')
    axs[1].set_title('Groundtruth Density map')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'dmap_' + name + '.png'))
    plt.close(fig)


def plotDensityMap(y_pred, y_real, folder, name, video_name):
    x = 1 + np.arange(len(y_pred))

    fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)

    extent = [x[0] - (x[1] - x[0]) / 2., x[-1] + (x[1] - x[0]) / 2., 0, 1]
    ax.imshow(y_pred[np.newaxis, :], cmap="Reds", aspect="auto", extent=extent)
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])
    ax.set_title('Predicted Density map')

    ax2.imshow(y_real[np.newaxis, :], cmap="Reds", aspect="auto", extent=extent)
    ax2.set_yticks([])
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_title('Ground truth Density map')

    # plt.title(video_name)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'dmap_' + name + '.png'))
    plt.close(fig)


def plotRepetitionCounts(y_true, y_pred):
    x = 1 + np.arange(len(y_pred))

    plt.figure(figsize=(20,20))

    plt.plot(x, y_pred, label='Predicted', marker='o')
    plt.plot(x, y_true, label='Ground truth', marker='o')

    plt.legend()

    plt.xlabel('Test Video index')
    plt.ylabel('Repetition Count')

    plt.savefig('repCountPlot.png')


def saveRepetitionCounts(video_paths, y_true, y_pred):
    results_df = pd.DataFrame(columns=['Video index', 'Video path', 'Predicted Count', 'Ground truth Count'])
    for i in range(len(video_paths)):
        newEntry = {'Video index': i + 1,
                    'Video path': video_paths[i][0],
                    'Predicted Count': y_pred[i],
                    'Ground truth Count': y_true[i]}
        #print(newEntry)
        results_df = results_df.append(newEntry, ignore_index=True)
    results_df.to_csv('repCounts.csv', index=False)


if __name__ == '__main__':
    f = open("agnostic_raw_results.pkl", "rb")
    results = pickle.load(f)

    video_paths = results['video_path']
    #print(len(video_paths))

    predicted_density_map = results['pred_dm']
    #print(len(predicted_density_map))

    ground_truth_density_map = results['actual_dm']
    #print(len(ground_truth_density_map))

    #print(predicted_density_map[0])
    #print(ground_truth_density_map[0])
    #
    for i in tqdm(range(len(ground_truth_density_map))[::10]):
           y_real = ground_truth_density_map[i][0]
           y_pred = (predicted_density_map[i][0]).astype(np.float32)
           plotDMSequence(y_pred, y_real, 'results', f'{i + 1}', video_paths[i])

    predicted_rep_counts = [round(sum(predicted_density_map[i][0])) for i in range(len(predicted_density_map))]
    #print(predicted_rep_counts[::15])
    print(predicted_rep_counts[::10])

    ground_truth_rep_counts = [round(sum(ground_truth_density_map[i][0])) for i in range(len(ground_truth_density_map))]
    #print(ground_truth_rep_counts[::15])
    print(ground_truth_rep_counts[::10])

    plotRepetitionCounts(y_true=ground_truth_rep_counts, y_pred=predicted_rep_counts)
    # saveRepetitionCounts(video_paths=video_paths, y_pred=predicted_rep_counts, y_true=ground_truth_rep_counts)
