import os
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 # The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";
import keras
from keras.callbacks import EarlyStopping, History
from keras import backend as K
from keras.layers import Input, Dense, Dropout, MaxPooling2D, Conv2D, Flatten, AveragePooling2D
from keras.models import Model, Sequential
from keras.utils import to_categorical
import numpy as np
from ovito.io import import_file, export_file
from collections import Counter
from os import listdir
from os.path import isfile
from keras.models import model_from_json
from itertools import product, combinations
from sklearn.model_selection import train_test_split

def run_convolutional(
        X_train, X_test, y_train, y_test, dumpname, history_dumpname,
        shape=(50, 50), epochs=100, batch_size=512, units=255):
    opt = Adam(learning_rate=0.001)
    history = History()
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1), history]
    pool_size = 2
    model = Sequential()
    model.add(Conv2D(16, 3, input_shape=(*shape, 1), activation='relu', padding='same'))
    model.add(AveragePooling2D(pool_size=pool_size))
    model.add(Conv2D(32, 3, input_shape=(*shape, 1), activation='relu', padding='same'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(AveragePooling2D(pool_size=pool_size))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(AveragePooling2D(pool_size=pool_size))
    model.add(Flatten())
    # model.add(Dropout(0.3))
    model.add(Dense(units, activation='softmax'))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(
        X_train, y_train, epochs=epochs,
        batch_size=batch_size, shuffle=True,
        callbacks=callbacks,
        validation_data=(X_test, y_test))
    model_json = model.to_json()
    with open(dumpname+'.json', "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(f"{dumpname}.h5")
            print("Saved model to disk")
    np.save(history_dumpname, history.history)
    return model

def test_model(model, test_data, tol=0.01):
    #Test on real data
    prediction_proba = model.predict(test_data, verbose=1)
    prediction = np.argmax(prediction_proba, axis=1)
    #Remove low probability results
    other = -1
    best_proba = prediction_proba[np.arange(prediction_proba.shape[0]), prediction]
    for i, proba in enumerate(best_proba):
        if proba < tol:
            prediction[i] = other
    return prediction

def write_data(datafile, prediction, dumpname):
    #Dump to ovito file
    pipe = import_file(datafile)
    data = pipe.compute()
    data.particles_.create_property('Cluster', data=prediction)
    export_file(data, dumpname, 'lammps/dump', columns=['Particle Type', 'Position.X', 'Position.Y', 'Position.Z', 'Cluster'])

def make_file(directory, dumpdir, exclude=' ', neighbors=' '):
    files = [f for f in listdir(directory) if not f.endswith(exclude) and not f'nneigh{neighbors}' in f]
    dumpfiles = [dumpdir+f.split('/')[-1] for f in files]
    data_files = [directory+f for f in files]
    return data_files, dumpfiles


def map_to_labels(prediction, label_dict, keep_biggest=None):
    #Make dict of labels from -1 to length of label_count
    label_count = Counter(prediction) 
    print('Label_count', label_count)
    print('Label_dict', label_dict)
    if keep_biggest:
        if -1 in dict(label_count.most_common(keep_biggest)).keys():
            label_count = dict(label_count.most_common(keep_biggest+1))
        else:
            label_count = dict(label_count.most_common(keep_biggest))

    try: del label_count[-1] #Remove disordered
    except KeyError:
        pass
    compressed_labels = np.arange(len(label_count), 0, -1) #Labels for mapping
    compressed_dict = dict(zip(label_count.keys(), compressed_labels))
    compressed_dict[-1] = -1
    print(compressed_dict)
    mapping = {} 
    for name, label in label_dict.items():
        for key, value in compressed_dict.items():
            if key == label:
                mapping[name] = value

    #Remove None
    compressed_prediction = []
    for pred in map(compressed_dict.get, prediction):
        if pred == None:
            compressed_prediction.append(-1)
        else:
            compressed_prediction.append(pred)
    return compressed_prediction, mapping

def train_test(data, shuffle=True, percent_training_data=0.8):
    if shuffle: np.random.shuffle(data)
    if data.ndim > 2:
        print(f'Data shape is {data.ndim}, raveling down to 2 dims')
        data = data.reshape(len(data), np.prod(data.shape[1:]))
    #Number of traning samples
    N_training_samples = int(np.floor(percent_training_data*data.shape[0]))
    #Devide into train and test sets
    data_train = data[:N_training_samples]
    data_test = data[N_training_samples:]
    #Reshape data
    data_train = data_train.reshape(len(data_train), np.prod(data_train.shape[1:]))
    data_test = data_test.reshape(len(data_test), np.prod(data_test.shape[1:]))
    data_train = data_train
    data_test = data_test
    data = data
    return data_train, data_test


def model_exists(model_dumpname):
    return isfile(model_dumpname+'.json')

if __name__=='__main__':
    # training_data = np.load('datasets/crystal_dataset_nrandomNone_startNone_end6.npy')
    # crystal_labels = np.load('datasets/crystal_labels_nrandomNone_startNone_end6.npy', allow_pickle=True).item()
    training_data = np.load('datasets/methane_dataset_nrandomNone_startNone_endNone.npy')
    crystal_labels = np.load('datasets/methane_labels_nrandomNone_startNone_endNone.npy', allow_pickle=True).item()
    print(crystal_labels)
    # training_data = np.load('datasets/crystal_dataset_nrandom5.npy')
    # crystal_labels = np.load('datasets/crystal_labels_nrandom5.npy', allow_pickle=True).item()
    # StandardScaler(copy=False).fit_transform(training_data[:, :-1])

    #Split data
    training_data[0][0] = 100
    print(training_data[0][0])
    X_train, X_test = train_test(training_data, percent_training_data=0.8, shuffle=True)
    print(X_train.dtype)
    print(X_train.shape, X_test.shape)
    print(X_train[0][0])

    exit()
    X_train, X_test = train_test_split(training_data, shuffle=True, train_size=0.8)
    del training_data
    y_train, y_test = to_categorical(X_train[:, -1]), to_categorical(X_test[:, -1])
    X_train, X_test = X_train[:, :-1], X_test[:, :-1]
    print(X_train.shape)
    shape = [int(np.sqrt(X_train.shape[-1]))]*2
    X_train = X_train.reshape(-1, *shape, 1)
    X_test = X_test.reshape(-1, *shape, 1)
    print(X_train.shape)

    #Testdata dirs
    quasidata_dir = '/home/kristtuv/Documents/master/src/python/allquasidata/'
    quasidata_adjacency_dir = 'quasicrystal_files_adjacency/'
    methanedata_dir = '/home/kristtuv/Documents/master/src/python/Grace_ase/datafiles/' 
    methanedata_adjacency_dir = '/home/kristtuv/Documents/master/src/python/Grace_ase/dumpfiles/adjacency/' 
    results_dumpdir = 'results/'
    model_evaluation_dumpdir = 'model_evaluation_conv/'
    model_dumpdir = 'nn_models_conv/'
    dumpname = 'methane_dataset_conv'

    # adjacency_files, _ = make_file(quasidata_adjacency_dir, model_dumpdir, neighbors=30)
    # data_files, dump_names = make_file(quasidata_dir, results_dumpdir, exclude='.log', neighbors=30)
    # adjacency_files, data_files, dump_names = sorted(adjacency_files), sorted(data_files), sorted(dump_names)
    # print(*zip(adjacency_files, data_files, dump_names), sep='\n')
    adjacency_files, _ = make_file(methanedata_adjacency_dir, model_dumpdir, neighbors=30)
    data_files, dump_names = make_file(methanedata_dir, results_dumpdir, exclude='.log', neighbors=30)
    adjacency_files, data_files, dump_names = sorted(adjacency_files), sorted(data_files), sorted(dump_names)
    a, df, dn = [], [], []
    for i, j, k in zip(adjacency_files, data_files, dump_names):
        if '156' in i:
            a.append(i)
        if '156' in j:
            df.append(j)
        if '156' in k:
            dn.append(k)
    adjacency_files, data_files, dump_names = a, df, dn

    batch_size=512
    run_predict = True
    recompute_model = False
    model_dumpname = (
            model_dumpdir
            +dumpname)
    model_evaluation_dumpname = (
            model_evaluation_dumpdir
            +dumpname)
    history_dumpname = (
            model_evaluation_dumpdir
            +dumpname
            +'_history')

    if model_exists(model_dumpname) and not recompute_model:
        print('======Model exists======')
        with open(model_dumpname+'.json', 'r') as json_file:
            loaded_json_file = json_file.read()
            model = model_from_json(loaded_json_file)
            model.load_weights(model_dumpname+'.h5')
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print('=======Calculating Model========')
        model = run_convolutional(
                X_train, X_test, y_train, y_test, model_dumpname,
                history_dumpname, shape=shape, epochs=5, batch_size=batch_size,
                units=len(crystal_labels))
        evaluation = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
        print('EVALUATION: ', evaluation)
    if run_predict:
        for datafile, adjacencyfile, dumpname in zip(data_files, adjacency_files, dump_names):
            data = np.load(adjacencyfile)
            data = data.reshape(-1, np.prod(data.shape[1:]))
            # StandardScaler(copy=False).fit_transform(data)
            data = data.reshape(-1, *shape, 1)
            print(data.shape)
            prediction = test_model(model, data, tol=0.8)
            print(np.unique(prediction))
            np.save('prediction_testing', prediction)
            prediction, labels = map_to_labels(prediction, crystal_labels, keep_biggest=None)
            np.save(dumpname+'_clusters', Counter(prediction))
            np.save(dumpname+'_labels', labels)
            max_prediction = Counter(prediction)
            write_data(datafile, prediction, dumpname)
