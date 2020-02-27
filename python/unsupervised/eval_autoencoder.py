import numpy as np
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from util import is_dir
def walk_path(path, endswith='evaluation.npy'):
    for root, dirs, files in os.walk(path):
        if files:
            for name in files:
                if name.startswith('restart_step_1530000') and name.endswith(endswith):
                    yield os.path.join(root, name)
def plot_precision(files_by_power, adjacencies, save=False, show=True):
    fig, ax = plt.subplots()
    for power, files in files_by_power.items():
        recalls_dump = f'recall/recalls{power}'
        recalls = []
        neighbors = []
        files = sorted(files)
        for f in files:
            model = f.split('.npy', 1)[0].replace('./', '')
            for adjacency in adjacencies:
                if model in adjacency:
                    nneigh = re.search('nneigh(\d\d)', f).group(1)
                    print()
                    print('Power ', power)
                    print('Neighbors', nneigh)
                    recall_dump = recalls_dump + '/'
                    is_dir(recall_dump)
                    recall_dump = recall_dump + f'recall_power{power}_nneigh{nneigh}'
                    if os.path.isfile(recall_dump+'.npy'):
                        print(f'File {recall_dump} exists')
                        _, nneigh, recall =  np.load(recall_dump+'.npy')
                    else:
                        y_true = np.load(adjacency).ravel()
                        y_score = np.load(f).ravel()
                        y_score = np.where(y_score>0.5, 1, 0)
                        recall = recall_score(y_true, y_score)
                        np.save(recall_dump, [power, nneigh, recall])
                    print('Neighbors ', nneigh)
                    print('Recall ', recall)
                    recalls.append(float(recall))
                    neighbors.append(int(nneigh))
        # np.save(recalls_dump, [power, neighbors, recalls])
        print('@@@@@@@@@@@@')
        print('Recalls :' , recalls)
        print('Neighbors :' , neighbors)
        print('@@@@@@@@@@@@')


        if isinstance(power, int):
            p = str(2**power)
            ax.plot(neighbors, recalls, label=f'Latent Dim {p}')
        elif isinstance(power, str):
            ax.plot(neighbors, recalls, label=f'Layers {power}')
        ax.set_xlabel('Number of neighbors')
        ax.set_ylabel('Recall')

    plt.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels)

    top = '/home/kristtuv/Documents/master/latex/plots/'
    if save:
        plt.savefig(f'{top}recall.png', transparent=True)
    if show:
        plt.show()
    plt.close()

def plot_power(files_by_power, metric='test_acc', save=False, show=True):
    fig, ax = plt.subplots()
    for power, files in files_by_power.items():
        files = sorted(files)
        evaluation = []
        neighbors = []
        for f in files:
            nneigh = re.search('nneigh(\d\d)', f).group(1)
            neighbors.append(int(nneigh))
            train_loss, train_acc, test_loss, test_acc = np.load(f)
            if metric=='test_acc': 
                ylabel = 'Test Accuracy'
                evaluation.append(test_acc)
            if metric=='test_loss': 
                ylabel = 'Test Loss'
                evaluation.append(test_loss)
            if metric=='train_acc': 
                ylabel = 'Training Accuracy'
                evaluation.append(train_acc)
            if metric=='train_loss': 
                ylabel = 'Training Loss'
                evaluation.append(train_loss)

        if isinstance(power, int):
            p = str(2**power)
            ax.plot(neighbors, evaluation, label=f'Latent Dim {p}')
        elif isinstance(power, str):
            ax.plot(neighbors, evaluation, label=f'Layers {power}')

        ax.set_xlabel('Number of neighbors')
        ax.set_ylabel(ylabel)
    plt.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels)
    top = '/home/kristtuv/Documents/master/latex/plots/'
    if save:
        plt.savefig(f'{top}autoencoderByPower.png', transparent=True)
    if show:
        plt.show()

def plot_neighbor(files_by_neighbor):
    fig, ax = plt.subplots()
    for neighbor, files in files_by_neighbor.items():
        files = sorted(files)
        metric = []
        powers = []
        for f in files:
            power = re.search('autoconstructpowerstart(\d)', f).group(1)
            powers.append(int(power))
            train_loss, train_acc, test_loss, test_acc = np.load(f)
            metric.append(test_acc)

        ax.plot(powers, metric, label=neighbor)

    plt.legend()
    # plt.savefig('autoencoderByNeighbor.png', transparent=True)
    plt.show()

def get_evaluation():
    files_by_power = defaultdict(list)
    files_by_neighbor = defaultdict(list)
    for f in walk_path('.'):
        # print(f)
        try:
            powerstart = int(re.search('autoconstructpowerstart(\d)', f).group(1))
        except AttributeError as e:
            powerstart = re.search('layers(.+)_act', f).group(1)

        neighbor = re.search('nneigh(\d\d)', f).group(1)
        files_by_power[powerstart].append(f)
        files_by_neighbor[int(neighbor)].append(f)
    return files_by_power, files_by_neighbor

def get_autoencoder():
    files_by_power = defaultdict(list)
    files_by_neighbor = defaultdict(list)
    for f in walk_path('.', endswith='autoencoder.npy'):
        try:
            powerstart = int(re.search('autoconstructpowerstart(\d)', f).group(1))
        except AttributeError as e:
            powerstart = re.search('layers(.+)_act', f).group(1)

        neighbor = re.search('nneigh(\d\d)', f).group(1)
        files_by_power[powerstart].append(f)
        files_by_neighbor[int(neighbor)].append(f)
    return files_by_power, files_by_neighbor

def get_adjacency():
    path = '../dumpfiles/adjacency/correct/'
    adjacency = []
    for f in walk_path(path, endswith=''):
        adjacency.append(f)
    return adjacency



files_by_power, _ = get_autoencoder()
adjacency = get_adjacency()
plot_precision(files_by_power, adjacency, save=True, show=True)
# print(files_by_power)
exit()

# plot_neighbor(files_by_neighbor)
plot_power(files_by_power, metric='test_acc', save=True, show=True)




