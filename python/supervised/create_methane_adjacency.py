from adjacency import Adjacency
from ovito.io import import_file
import ase
import os
from util import is_dir
from ovito.modifiers import CoordinationAnalysisModifier
import numpy as np
import re
from os import listdir
import pathlib
from pathlib import PurePath

def file_handling(filedir, dumpdir):
    is_dir(dumpdir)
    files = [f for f in listdir(filedir) if not f.endswith('.log')]
    files = [PurePath(p, f[0]) for p, _, f in os.walk(filedir) if f]
    dumpfiles = [PurePath(dumpdir,f.parts[9]+f.name) for f in files]
    return files, dumpfiles

if __name__=='__main__':
    # filedir = '/home/kristtuv/Documents/master/src/python/Grace_ase/datafiles/'
    filedir = '/home/kristtuv/Documents/master/src/python/polycrystalline_shear_snapshots/'

    dumpdir = 'methane_files_adjacency/' 
    files, dumpfiles = file_handling(filedir, dumpdir)

    number_of_bins = 100
    coordination_cutoff = 10
    coordination = CoordinationAnalysisModifier(
            cutoff=coordination_cutoff,
            number_of_bins=number_of_bins
            )
    # num_neighbors = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_neighbors = [50]
    for n in num_neighbors:
        for f, d in zip(files, dumpfiles):
            adjacency = Adjacency([str(f)])
            adj, data = adjacency.build_adjacency(coordination, 0, [str(d)], num_neighbors=n, recompute=True)
    
