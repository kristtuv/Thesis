import numpy as np
import operator
import os
from os import listdir
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from collections import Counter
from itertools import product
import re
from util import is_dir

def map_array_to_index(array, return_mapping=True):
    #Map array to continuous array
    unique = sorted(np.unique(array))
    idx = np.arange(min(unique), len(unique)+min(unique))
    mapping = dict(zip(idx, unique))
    for i, element in enumerate(unique):
        find_elements = np.where(array==element)
        array[find_elements] = idx[i]
    if return_mapping:
        return array, mapping
    else:
        return array

def cond(x):
    #Sort condition
    k = float(re.search('_k(\d\.\d+)_', x).group(1))
    phi = float(re.search('_phi(\d\.\d+)_', x).group(1))
    return (k, phi)

def walk_path(directory, contains='', endswith='labels.npy'):
    paths = []
    for root, dirs, files in os.walk(directory):
        # print(dirs)
        # continue
        if files and str(contains) in root:
            for name in files:
                if name.endswith(endswith):
                    path = os.path.join(root, name)
                    paths.append(path)
    return paths



def make_plot(neighbors=10, filedir='results_evaluated/', dumpdir='', save=False, most_common=1, fullname=True, show=False, remove_noise=False):
    #Import pearson
    pearson = np.load('utility/pearson_symbols.npy', allow_pickle=True).item()
    pearson['methane_hydrate_I'] = 'ClaI'
    pearson['methane_hydrate_II'] = 'ClaII'
    pearson['methane_hydrate_H'] = 'ClaIV'
    pearson['disordered'] = 'disordered' 

    #Import files
    cluster_files = sorted(walk_path(filedir, contains='numneighbors'+str(neighbors), endswith='clusters.npy'), key=cond)
    label_files = sorted(walk_path(filedir, contains='numneighbors'+str(neighbors), endswith='labels.npy'), key=cond)
    ks = [float(re.search('_k(\d\.\d+)_', f).group(1)) for f in cluster_files]
    phis = [float(re.search('_phi(\d\.\d+)_', f).group(1)) for f in cluster_files]


    #Create mesh
    x = np.unique(ks)
    y = np.unique(phis)
    x = np.pad(x, (0, 1), constant_values=(2*x[-1]-x[-2]))
    y = np.pad(y, (0, 1), constant_values=(2*y[-1]-y[-2]))
    X, Y = np.meshgrid(x, y)

    Z = np.zeros((len(x), len(y)))
    unique_pearson = {}
    for c, l, phi, k in zip(cluster_files, label_files, phis, ks):
        #Set Z index
        j = np.where(x == k)
        i = np.where(y == phi)

        labels = np.load(l, allow_pickle=True).item() 
        labels['disordered'] = -1
        labels = dict(zip(labels.values(), labels.keys()))
        clusters = np.load(c, allow_pickle=True).item() 
        cluster_all = clusters.most_common()
        # print(cluster_all)
        if remove_noise:
            try:
                if cluster_all[0][0] == -1:
                    cluster_max = cluster_all[1][0]
                else:
                    cluster_max = cluster_all[0][0]
            except:
                cluster_max = cluster_all[0][0]

        else:
            try:
                cluster_max = clusters.most_common(most_common)[most_common-1][0]
                print(cluster_max)
            except:
                cluster_max = clusters.most_common(1)[0][0]

        if fullname:
            counts = list(map(lambda x: x[1],  cluster_all))
            map_name = list(map(lambda x: x[0], cluster_all))
            lab = [labels[m] for m in map_name]
            map_pearson = lab
            name = labels[cluster_max]
            pearson_name = name

        else:
            counts = list(map(lambda x: x[1],  cluster_all))
            map_name = list(map(lambda x: x[0], cluster_all))
            lab = [labels[m] for m in map_name]
            map_pearson = list(map(lambda x: pearson[x], lab))
            name = labels[cluster_max]
            pearson_name = pearson[name]
        # print(*zip(lab, counts), sep='\n')
        # print()

        
        if pearson_name in unique_pearson:
            Z[i, j] = unique_pearson[pearson_name]
        else:
            unique_pearson[pearson_name] = cluster_max
            Z[i, j] = cluster_max 

    pearson_names = sorted(unique_pearson.items(), key=operator.itemgetter(1))
    # print(pearson_names)
    # exit()
    sorted_pearson_names = [n[0] for n in pearson_names]
    Z[-1],Z[:, -1] = Z[-2], Z[:, -2]
    Z, mapping = map_array_to_index(Z)
    ticks = np.unique(Z).astype(int)
    bounds = [-1.5] + list(np.unique(Z).astype(int) + 0.5)
    cmap = plt.get_cmap('jet', len(bounds)-1)#, len(bounds)+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    # cax = ax.imshow(np.flip(Z.T, axis=0), cmap=cmap)
    cax = ax.pcolor(X, Y, Z, cmap = cmap, edgecolors='w')
    cbar = fig.colorbar(cax, norm=norm, boundaries=bounds, ticks=ticks)
    print(cbar)
    cbar.ax.set_yticklabels(sorted_pearson_names)
    plt.tight_layout()
    if save:
        plt.savefig(
                dumpdir+f'neighbors{neighbors}_mostcommon{most_common}_fullname{fullname}', transparent=True
                )
        
    if show:
        plt.show()
    plt.close()

# dumpdir = '../../../latex/plots/phase_tol09/'
dumpdir = '../../../latex/plots/appendix/'

# filedir = 'results_evaluated_tol09/results_dense/'
# network = 'dense'

# filedir = 'results_evaluated_tol09/results_conv_sweep/'
# filedir = 'results_evaluated_remove_one/results_remove_one/'
# filedir = 'results_evaluated_unit/results_unit/'
# filedir = 'results_evaluated_unit_tol08/results_unit/'
# filedir = 'results_evaluated_unit_tol03_newbreak/results_unit_newbreak/'
filedir = 'results_evaluated_unit_tol03/results_conv_sweep/'
network = 'conv'
is_dir(dumpdir)

# for n in range(10, 81, 10):
# for n in [10, 60, 80]:
for n in [60]:
    for m in range(1, 3):
    # for m in range(2, 3):
        for fullname in [True]:
        # for fullname in [True, False]:
            print(m)
            print(n)
            print(dumpdir)
            print(filedir)
            make_plot(
                    neighbors=n,
                    filedir=filedir,
                    dumpdir=dumpdir+network,
                    save=False,
                    show=True,
                    most_common=m,
                    fullname=fullname,
                    remove_noise=False)
