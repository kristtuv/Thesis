import os
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
from util import is_dir
import re
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
            'datasets/cutoff_finder/', 'datasets/get_first_minima_after_max/',
            'datasets/get_first_minima/', 'datasets/None/']

    files, labels = get_files(datadirs)
    files, labels = sorted(files, reverse=True), sorted(labels, reverse=True)
    dropouts = [0.1, 0.3, 0.5, 0.0]
    act_func = ['sigmoid', 'relu', 'tanh']
    activations = product(act_func, act_func, repeat=2)
    for training_data_name, crystal_labels_name in zip(files, labels):
        training_data = np.load(training_data_name).astype(np.int8)
        np.random.shuffle(training_data)
        X = training_data[:, :-1]
        X = X.reshape(-1, np.prod(X.shape[1:]))
        y = to_categorical(training_data[:, -1])
        for dropout in dropouts:
            for act in activations:
                numneighbors = re.search('numneighbors(\d+)', training_data_name).group(1)
                crystal_labels = np.load(crystal_labels_name, allow_pickle=True).item()
                shape = training_data.shape[-1]-1
                shape_1 = int(shape/2)
                shape_2 = int(shape/4)
                shape_3 = int(shape/6)
                shape_4 = int(shape/8)
                #Set layer parameters
                layer_params = [
                        {'units':shape_1, 'input_shape': (shape,), 'activation':act[0]},
                        {'units':shape_2, 'activation':act[1]},
                        {'units':shape_3, 'activation':act[2]},
                        {'units':shape_4, 'activation':act[3]},
                        ]

                #Set fit parameters
                fit_params = {'epochs': 150, 'batch_size': 512}
                model_params = {'dropout':True, 'dropout_rate': dropout}


                #Shape data

                #Dump directoies
                training_data_file_name = training_data_name.split('/')[-1]
                subdir = training_data_file_name.replace('.npy', '/')
                results_dumpdir = 'results_dense/'+subdir
                is_dir(results_dumpdir)

                #Dumpnames
                layer_dumpname = params_to_filename(layer_params)
                fit_dumpname = params_to_filename({**fit_params, **model_params})
                results_dumpdir = (
                   results_dumpdir
                   + 'model'
                   +layer_dumpname
                   +fit_dumpname
                   +'/')
                model_dumpname = results_dumpdir + 'model'
                history_dumpname = results_dumpdir + 'history'
                is_dir(results_dumpdir)

                recompute_model=False
                run_predict=False
                if model_exists(model_dumpname) and not recompute_model:
                   print('======Model exists======')
                   print(model_dumpname)
                   with open(model_dumpname+'.json', 'r') as json_file:
                       loaded_json_file = json_file.read()
                       model = model_from_json(loaded_json_file)
                       model.load_weights(model_dumpname+'.h5')
                       model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                else:
                    print('=======Calculating Model========')
                    print(model_dumpname)
                    dense_model = DenseNet(num_classes=len(crystal_labels),
                            layer_params=layer_params,
                            **model_params)
                    dense_model.fit(X, y, **fit_params)
                    dense_model.save_model(model_dumpname, history_dumpname)
                    model = dense_model.model

                if run_predict:
                    #Choose test_data
                    quasicrystal = True
                    methane = False
                    if quasicrystal:
                        quasidata_dir = '/home/kristtuv/Documents/master/src/python/allquasidata/'
                        quasidata_adjacency_dir = 'quasicrystal_files_adjacency/'
                        adjacency_files, _ = make_file(quasidata_adjacency_dir, results_dumpdir, keep=['nneigh'+numneighbors])
                        data_files, dump_names = make_file(quasidata_dir, results_dumpdir, exclusions=['.log'])

                    else:
                        methanedata_dir = '/home/kristtuv/Documents/master/src/python/Grace_ase/datafiles/' 
                        methanedata_adjacency_dir = '/home/kristtuv/Documents/master/src/python/Grace_ase/dumpfiles/adjacency/' 
                        adjacency_files, _ = make_file(
                                methanedata_adjacency_dir, model_dumpdir,
                                exclusions=['nneigh30', 'water'], keep=['156'])
                        data_files, dump_names = make_file(
                                methanedata_dir, results_dumpdir,
                                exclusions=['water'], keep=['156'])
                    adjacency_files, data_files, dump_names = sorted(adjacency_files), sorted(data_files), sorted(dump_names)
                    for datafile, adjacencyfile, dumpname in zip(data_files, adjacency_files, dump_names):
                       data = np.load(adjacencyfile)
                       data = data.reshape(-1, np.prod(data.shape[1:]))
                       prediction = predict(model, data, tol=0.5)

                       np.save(dumpname+'prediction_testing', prediction)
                       # prediction, crystal_labels = map_to_labels(prediction, crystal_labels, keep_biggest=None)
                       crystal_labels['disordered'] = -1
                       print(crystal_labels)
                       print(Counter(prediction))
                       np.save(dumpname+'_clusters', Counter(prediction))
                       np.save(dumpname+'_labels', crystal_labels)
                       write_data(datafile, prediction, dumpname)
