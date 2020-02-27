import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";
import keras
import numpy as np

from collections import Counter, defaultdict
from itertools import product, combinations
from keras.callbacks import EarlyStopping, History
from keras import backend as K
from keras. layers import BatchNormalization
from keras.layers import Input, Dense, Dropout, Conv2D, AveragePooling2D, Flatten
from keras.models import Model, model_from_json, Sequential
from keras.utils import to_categorical
from os import listdir
from os.path import isfile
from ovito.io import import_file, export_file
from sklearn.preprocessing import StandardScaler
def params_to_filename(layer_params):
    if not isinstance(layer_params, list):
        layer_params = [layer_params]

    #Collect dictionaries to one
    collect_parameters = defaultdict(list)
    for d in layer_params:
        for k, v in d.items():
            collect_parameters[str(k).replace('_', '')].append(str(v))

    #Unpack dictionary to string
    filename = ''
    for k, v in collect_parameters.items():
        filename +='_'+str(k)+'|'.join(v)

    #Remove unwnated characters
    remove = str.maketrans(' ', '|', '(),')
    filename = filename.translate(remove)
    return filename

def make_file(directory, dumpdir, exclusions=[], keep=[]):
    files = listdir(directory) 
    if keep:
        for k in keep:
            files = [f for f in files if k in f]
    if exclusions:
        for exclusion in exclusions:
            files = [f for f in files if exclusion not in f]
    dumpfiles = [dumpdir+f.split('/')[-1] for f in files]
    data_files = [directory+f for f in files]
    return data_files, dumpfiles

def model_exists(model_dumpname):
    return isfile(model_dumpname+'.json')

def write_data(datafile, prediction, dumpname, frame=-1):
    pipe = import_file(datafile)
    data = pipe.compute(frame)
    data.particles_.create_property('Cluster', data=prediction)
    export_file(data, dumpname, 'lammps/dump', columns=['Particle Type', 'Position.X', 'Position.Y', 'Position.Z', 'Cluster'])

def predict(model, X_test, tol=0.01):
    prediction_proba = model.predict(X_test, verbose=1)
    prediction = np.argmax(prediction_proba, axis=1)
    other = -1
    best_proba = prediction_proba[np.arange(prediction_proba.shape[0]), prediction]
    for i, proba in enumerate(best_proba):
        if proba < tol:
            prediction[i] = other
    return prediction


def map_to_labels(prediction, label_dict, keep_biggest=None):
    #Make dict of labels from -1 to length of label_count
    label_count = Counter(prediction) 
    if keep_biggest:
        if -1 in dict(label_count.most_common(keep_biggest)).keys():
            label_count = dict(label_count.most_common(keep_biggest+1))
        else:
            label_count = dict(label_count.most_common(keep_biggest))

    try: del label_count[-1] #Remove disordered
    except KeyError:
        pass
    compressed_labels = np.arange(len(label_count)-1, -1, -1) #Labels for mapping
    compressed_dict = dict(zip(label_count.keys(), compressed_labels))
    compressed_dict[-1] = -1
    mapping = {} 
    for name, label in label_dict.items():
        for key, value in compressed_dict.items():
            if key == label:
                mapping[name] = value
    mapping['disordered'] = -1

    #Remove None
    compressed_prediction = []
    for pred in map(compressed_dict.get, prediction):
        if pred is None:
            compressed_prediction.append(-1)
        else:
            compressed_prediction.append(pred)
    return compressed_prediction, mapping

class NeuralNet:
    def __init__(self, patience=10):
        self.history = History()
        self.callbacks = [
                EarlyStopping(monitor='val_loss', patience=patience, verbose=1), self.history
                ]

    def save_model(self, model_dumpname, history_dumpname):
        model_json = self.model.to_json()
        with open(model_dumpname+'.json', "w") as json_file:
                json_file.write(model_json)
                self.model.save_weights(f"{model_dumpname}.h5")
                print("Saved model to disk")
        np.save(history_dumpname, self.history.history)
        print("Saved history to disk")

    def fit(
        self, X, y, epochs=100, batch_size=512, validation_split=0.4,
        ):
        self.model.fit(
            X, y, epochs=epochs,
            validation_split=validation_split,
            batch_size=batch_size, shuffle=True,
            callbacks=self.callbacks,
            # use_multiprocessing=True
            )

class ConvNet(NeuralNet):
    def __init__(self, num_classes=255, layers='caccaca', dropout=0.1, layer_params=None):
        super().__init__()
        layer_names = {'c': Conv2D, 'a':AveragePooling2D}
        model = Sequential()
        for i, layer in enumerate(layers):
            model.add(layer_names[layer](**layer_params[i]))
            if dropout and layer=='a':
                model.add(Dropout(dropout))
        model.add(Flatten())
        if dropout:
            model.add(Dropout(dropout))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        self.model=model



class DenseNet(NeuralNet):
    def __init__(self, num_classes=255, dropout_rate = 0.1, dropout=True, layer_params=[]):
        super().__init__()
        model = Sequential()
        for layer in layer_params:
            model.add(Dense(**layer))
            if dropout:
                model.add(Dropout(dropout_rate))
            print(model)
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model=model

