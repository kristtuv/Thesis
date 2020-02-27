import keras
from os import listdir
from keras.applications.resnet50 import ResNet50
import re
import numpy as np
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.utils import to_categorical


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
    files, labels = sorted(files), sorted(labels)
    for training_data_name, crystal_labels_name in zip(files, labels):
        numneighbors = re.search('numneighbors(\d+)', training_data_name).group(1)
        training_data = np.load(training_data_name).astype(np.int8)
        crystal_labels = np.load(crystal_labels_name, allow_pickle=True).item()
        shape = int(np.sqrt(training_data.shape[-1]-1))
        num_classes=len(crystal_labels)
        print(shape)
        np.random.shuffle(training_data)
        X = training_data[:, :-1]
        X = X.reshape(-1, shape, shape, 1)
        y = to_categorical(training_data[:, -1])
        try:
            base_model = ResNet50(include_top=False, input_shape=(shape, shape, 3))
            base_model.summary()
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X, y)

            exit()
        except ValueError as e:
            print(e)
