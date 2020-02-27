from adjacency import Adjacency
from ovito.io import import_file
import ase
import os
from util import is_dir
from ovito.modifiers import CoordinationAnalysisModifier
import numpy as np
import re
from os import listdir

def file_handling(filedir, dumpdir):
    is_dir(dumpdir)
    files = [f for f in listdir(filedir) if not f.endswith('.log')]

    dumpfiles = [dumpdir + f for f in files]
    files = [filedir+f for f in files]
    return files, dumpfiles

if __name__=='__main__':
    filedir = '/home/kristtuv/Documents/master/src/python/quasi/'
    dumpdir = 'quasicrystal_files_adjacency/'
    files, dumpfiles = file_handling(filedir, dumpdir)

    number_of_bins = 100
    coordination_cutoff = 10
    coordination = CoordinationAnalysisModifier(
            cutoff=coordination_cutoff,
            number_of_bins=number_of_bins
            )
    adjacency = Adjacency(files)
    num_neighbors = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cutoffs = ['cutoff_finder']
    for cutoff in cutoffs:
        dumpfiles = [d.replace('/', f'/{cutoff}/') for d in dumpfiles]
        for n in num_neighbors:
            all_cutoff = [cutoff]*len(files)
            adj, data = adjacency.build_adjacency(
                    coordination, -1, dumpfiles, num_neighbors=n,
                    inner_cutoff=all_cutoff, recompute=False
                    )
        
