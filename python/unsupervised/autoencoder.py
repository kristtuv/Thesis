import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras import regularizers
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import os

# def auroc(y_true, y_pred):
#     return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

class AutoEncoder:
    def __init__(self, epoch, frame, dump_filename):
        self.epoch = epoch
        self.data_train = None
        self.data_test = None
        self.dump_filename = dump_filename + f'_frame{frame}'

    def train_test(self, data, percent_training_data=0.6):
        if data.ndim > 2:
            print(f'Data shape is {data.ndim}, raveling down to 2 dims')
            data = data.reshape(len(data), np.prod(data.shape[1:]))

        np.random.shuffle(data)
        #Number of traning samples
        N_training_samples = int(np.floor(percent_training_data*data.shape[0]))
        print(N_training_samples)
        #Devide into train and test sets
        data_train = data[:N_training_samples]
        data_test = data[N_training_samples:]
        #Reshape data
        data_train = data_train.reshape(len(data_train), np.prod(data_train.shape[1:]))
        data_test = data_test.reshape(len(data_test), np.prod(data_test.shape[1:]))
        self.data_train = data_train
        self.data_test = data_test
        self.data = data
        return data_train, data_test

    def autoconstruct_layers(self, N_input_features, power_start=4):
        bases = np.arange(15)
        powers = np.power(2, bases)[power_start:]
        startunit = np.argmax(powers > N_input_features) - 2
        print(N_input_features)
        print(startunit)
        print(powers)
        input_img= Input(shape=(N_input_features,))
        encoded = Dense(units=powers[startunit], activation='relu')(input_img)
        for i in reversed( range(startunit)):
            encoded = Dense(units=powers[i], activation='relu')(encoded)
        decoded = Dense(units=powers[1], activation='relu')(encoded)
        for i in range(startunit+1):
            decoded = Dense(units=powers[i], activation='relu')(decoded)
        decoded = Dense(units=N_input_features, activation='relu')(decoded)
        return input_img, encoded, decoded


    def construct_layers(self, N_input_features, layers:list, activations=None):
        if activations==None:
            activations=['relu']*len(layers)
        input_img = Input(shape=(N_input_features,))
        encoded = Dense(
                units=layers[0], activation=activations[0],
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)
                )(input_img)
        for layer, activation in zip(layers[1:], activations[1:]):
            encoded = Dense(
                    units=layer, activation=activation,
                    kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01)
                     )(encoded)
        decoded = Dense(
                units=layers[-2], activation=activations[-2],
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)
                )(encoded)
        for layer, activation in zip(layers[-3::-1], activations[-3::-1]):
            decoded = Dense(
                    units=layer, activation=activation,
                    kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01)
                    )(decoded)
        decoded = Dense(units=N_input_features, activation='relu')(decoded)
        return input_img, encoded, decoded

    def autoencoder(
                self, data_train=None, data_test=None, autoconstruct_layers=True,
                power_start=4, input_layers=None, activations=None,
                recompute=False, plot_roc=False
                ):
            if input_layers is None:
                self.dump_filename = self.dump_filename + f'_autoconstructpowerstart{power_start}.npy'
            else:
                layers = '|'.join(map(str, input_layers))
                act = '|'.join(activations)
                self.dump_filename = self.dump_filename + f'_layers{layers}_act{act}.npy'

            if os.path.isfile(self.dump_filename) and not recompute:
                print(f'Precalculated: Loading {self.dump_filename}')
                encoded_imgs = np.load(self.dump_filename)
                return encoded_imgs, self.dump_filename
            else:
                if self.data_train is not None and self.data_test is not None:
                    data_train, data_test = self.data_train, self.data_test
                N_input_features = data_train.shape[1]
                if input_layers==None:
                    input_img, encoded, decoded = self.autoconstruct_layers(N_input_features, power_start)
                else:
                    input_img, encoded, decoded = self.construct_layers(
                            N_input_features, input_layers, activations)
                autoencoder=Model(input_img, decoded)
                encoder = Model(input_img, encoded)
                autoencoder.summary()
                autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
                autoencoder.fit(data_train, data_train,
                                epochs=self.epoch,
                                callbacks=callbacks,
                                batch_size=256,
                                shuffle=True,
                                validation_data=(data_test, data_test))

                evaluation_train = autoencoder.evaluate(data_train, data_train)
                evaluation_test = autoencoder.evaluate(data_test, data_test)
                evaluation = evaluation_train + evaluation_test
                encoded_imgs = encoder.predict(self.data, verbose=1)
                autoencoder_imgs = autoencoder.predict(self.data, verbose=1)
                print(autoencoder_imgs)
                print(f'Creating file {self.dump_filename}')
                np.save(self.dump_filename+'_evaluation', evaluation)
                np.save(self.dump_filename, encoded_imgs)
                np.save(self.dump_filename+'_autoencoder', autoencoder_imgs)
                return encoded_imgs, self.dump_filename

if __name__=='__main__':
    path = 'dumpfiles/adjacency/correct/'
    def walk_path(path):
        for root, dirs, files in os.walk(path):
            if files:
                for name in files:
                    if name.startswith('restart'):
                        yield os.path.join(root, name)

    layers = [[1500, 1000, 500, 256], [1500, 1000, 500], [1500, 1000]]
    for f in walk_path(path):
        print(f)
    for f in walk_path(path):
        dump = f.rsplit('/', 1)[-1]
        data = np.load(f)
        for power in range(3, 9):
            autoencoder = AutoEncoder(100, 0, f'autoencoder_test_relu/{dump}')
            autoencoder.train_test(data)
            autoencoder.autoencoder(power_start=power, plot_roc=True, recompute=False)

        for layer in layers:
            autoencoder = AutoEncoder(100, 0, f'autoencoder_test_relu/{dump}')
            autoencoder.train_test(data)
            activations = ['relu']*len(layer)
            autoencoder.autoencoder(input_layers=layer, activations=activations, plot_roc=True, recompute=False)




