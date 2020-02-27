import numpy as np
import os


def walk_path(path, endswith='evaluation.npy', startswith='restart'):
    for root, dirs, files in os.walk(path):
        if files:
            for name in files:
                if name.endswith(endswith) and name.startswith(startswith):
                    yield os.path.join(root, name)


d = 'dumpfiles/encoded/'
for f in walk_path(d):
    autoencoder = np.load(f)
    print(f, autoencoder)


