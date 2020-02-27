import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"]="0";
# from keras import backend as K
# import tensorflow as tf
# import tensorflow.python.keras.backend as K
# jobs = 10
# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=jobs,  
#                         inter_op_parallelism_threads=jobs, 
#                         allow_soft_placement=True, 
#                         device_count = {'CPU': jobs})
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)

import keras
from os import listdir
from nn import ConvNet, DenseNet
from nn import make_file, model_exists, params_to_filename, predict, map_to_labels, write_data
from keras.utils import to_categorical
from keras.models import model_from_json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import re
from util import is_dir
from itertools import product

def get_files(datadirs):
    all_files = []
    all_labels = []
    for d in datadirs:
        files = [d+f for f in listdir(d) if not 'labels' in f]
        labels = [d+l for l in listdir(d) if 'labels' in l]
        all_files.extend(files)
        all_labels.extend(labels)
    return all_files, all_labels


if __name__=='__main__':
    #Trainingdata
    datadirs = [
            # 'datasets_methane/cutoff_finder/']#, 'datasets/get_first_minima_after_max/',
            # 'datasets/cutoff_finder/']#, 'datasets/get_first_minima_after_max/',
            # 'datasets/get_first_minima/', 'datasets/None/']
            # 'datasets_remove_one/cutoff_finder/']
            'datasets_unit_newbreak/cutoff_finder/']

    files, labels = get_files(datadirs)
    files, labels = sorted(files, reverse=True), sorted(labels, reverse=True)
    # print(*zip(files, labels), sep='\n')
    # exit()
    act_func = ['relu', 'tanh']
    activations = list(product(act_func, act_func, repeat=2))
    kernel_sizes = [3, 5]
    dropouts=[0.3, 0.5, None]
    run_neighbors = [60]
    for training_data_name, crystal_labels_name in zip(files, labels):
        numneighbors = re.search('numneighbors(\d+)', training_data_name).group(1)
        if not int(numneighbors) in run_neighbors:
            continue
        training_data = np.load(training_data_name).astype(np.int8)
        crystal_labels = np.load(crystal_labels_name, allow_pickle=True).item()
        shape = int(np.sqrt(training_data.shape[-1]-1))
        #Shape data
        np.random.shuffle(training_data)
        X = training_data[:, :-1]
        X = X.reshape(-1, shape, shape, 1)
        y = to_categorical(training_data[:, -1])
        for dropout in dropouts:
            for act in activations:
                for kernel in kernel_sizes:
                    if int(numneighbors) <= 30:
                        layer_params = [
                        {'filters': 16, 'kernel_size': kernel, 'input_shape': (shape, shape, 1), 'padding':'same', 'activation':act[0]},
                        {'pool_size':2},
                        {'filters': 32, 'kernel_size': kernel, 'padding':'same', 'activation':act[1]},
                        {'filters': 64, 'kernel_size': kernel, 'activation':act[2], 'padding':'same'},
                        {'filters': 128, 'kernel_size': kernel, 'activation':act[3], 'padding':'same'},
                        ]
                        layers = 'caccc'

                    if int(numneighbors) > 30:
                        layer_params = [
                        {'filters': 16, 'kernel_size': kernel, 'input_shape': (shape, shape, 1), 'padding':'same', 'activation':act[0]},
                        {'pool_size':2},
                        {'filters': 32, 'kernel_size': kernel, 'padding':'same', 'activation':act[1]},
                        {'filters': 64, 'kernel_size': kernel, 'activation':act[2], 'padding':'same'},
                        {'pool_size':2},
                        {'filters': 128, 'kernel_size': kernel, 'activation':act[3], 'padding':'same'},
                        ]
                        layers = 'caccac'

                    fit_params = {'epochs': 150, 'batch_size': 512}


                    #Dump directoies
                    training_data_file_name = training_data_name.split('/')[-1]
                    subdir = training_data_file_name.replace('.npy', '/')
                    results_dumpdir = 'results_unit_newbreak/'+subdir
                    # results_dumpdir = 'results_conv_sweep/'+subdir
                    is_dir(results_dumpdir)

                    #Dumpnames
                    layer_dumpname = params_to_filename(layer_params)
                    fit_dumpname = params_to_filename({**fit_params})
                    results_dumpdir = (
                       results_dumpdir
                       + 'model'
                       +layer_dumpname
                       +fit_dumpname
                       +f'_dropout{dropout}'
                       +'/')
                    model_dumpname = results_dumpdir + 'model'
                    history_dumpname = results_dumpdir + 'history'
                    is_dir(results_dumpdir)

                    recompute_model=False
                    run_predict=False
                    if model_exists(model_dumpname) and not recompute_model:
                       print('======Model exists======')
                       print(model_dumpname)
                       print()
                       with open(model_dumpname+'.json', 'r') as json_file:
                           loaded_json_file = json_file.read()
                           model = model_from_json(loaded_json_file)
                           model.load_weights(model_dumpname+'.h5')
                           model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    else:
                        print('=======Calculating Model========')
                        conv_model = ConvNet(
                                num_classes=len(crystal_labels),
                                layer_params=layer_params,
                                dropout=dropout,
                                layers=layers)
                        conv_model.fit(X, y, **fit_params)
                        conv_model.save_model(model_dumpname, history_dumpname)
                        model = conv_model.model
        del X, y, training_data, crystal_labels

