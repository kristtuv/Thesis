# Import all the required Libraries
import tensorflow
import keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam

path = 'dumpfiles/adjacency/correct/'
for root, dirs, files in os.walk(path):
    if files:
        for name in files:
            if '20' in name:
                n = os.path.join(root, name)
                adj = np.load(n)
                adj = adj.reshape(*adj.shape, 1)
                #ENCODER
                inp = Input(adj.shape[1:])
                e = Conv2D(32, (3, 3), activation='relu')(inp)
                e = MaxPooling2D((2, 2))(e)
                e = Conv2D(64, (3, 3), activation='relu')(e)
                e = MaxPooling2D((2, 2))(e)
                e = Conv2D(64, (3, 3), activation='relu')(e)
                l = Flatten()(e)
                l = Dense(25, activation='softmax')(l)
                #DECODER
                d = Reshape((5,5,1))(l)
                d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(d)
                d = BatchNormalization()(d)
                d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(d)
                d = BatchNormalization()(d)
                d = Conv2DTranspose(32,(3, 3), activation='relu', padding='same')(d)
                decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)
                ae = Model(inp, decoded)
                ae.summary()


                # compile it using adam optimizer
                ae.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])#Train it by providing training images
                ae.fit(adj, adj, epochs=150, batch_size=256, verbose=1, shuffle=True, validation_split=0.2)
                np.save('conv_autotest', ae.predict(adj))

