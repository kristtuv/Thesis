import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from ovito.io import import_file
from os import listdir

d = 'crystal_files_unit/'
files = listdir(d)
pearson = np.load('pearson_symbols.npy', allow_pickle=True).item()
pearson['methane_hydrate_I'] = 'ClaI'
pearson['methane_hydrate_II'] = 'ClaII'
pearson['methane_hydrate_H'] = 'ClaIV'
pearson['a3bc'] = 'hP10'
pearson['a3b2'] = 'hP10'
keep = [
        'ClaI',
        'ClaII',
        'ClaIV',
        'cI16',
        'cP4',
        'cP8',
        'hP10',
        'hP2'
        ]


for f in files:
    f_strip = f.replace('.data', '')
    p = pearson[f_strip]
    if p in keep:
        pipe = import_file(d+f)
        data = pipe.compute()
        print(p, data.number_of_particles)
