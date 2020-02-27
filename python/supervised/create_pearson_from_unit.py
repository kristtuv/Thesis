from os import listdir
import numpy as np
from ovito.io import import_file, export_file
from collections import defaultdict

def map_name_to_pearson(datadir, pearson_dict, keep=['hP6'], n_keep=2):
    data_files = listdir(datadir)
    final = []
    n_keep_dict = defaultdict(int, zip(keep, np.zeros(len(keep), dtype=int)))
    for k in keep:
        for key, value in pearson_dict.items():
            if value == k and n_keep_dict[k] < n_keep:
                n_keep_dict[k] += 1
                for data in data_files:
                    if key in data:
                        print(k, data)
                        final.append(data)
    return list(np.unique(final))

        


if __name__=='__main__':
    datadir = 'crystal_files_unit/'
    data_files = listdir(datadir)
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

    results = map_name_to_pearson(data_files, pearson, keep=keep)
    print (results)

