import os
from scipy.signal import argrelmin
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
import numpy as np
import inspect
import matplotlib.pyplot as plt

def check_frame(frame, pipe):
    if frame < 0:
        return pipe.source.num_frames + frame + 1
    else:
        return frame

def is_dir(path):
    if not os.path.isdir(path):
        print(f'Making directory {path}')
        os.mkdir(path)

def cutoff_finder(data, plot=False):
    rdf = data.series['coordination-rdf'].as_table()
    max_height = np.max(rdf[:, 1])
    peaks = find_peaks(rdf[:, 1], height=max_height*0.1)
    extrema = peaks[0][0]
    rdf_from_first_peak = rdf[extrema:]
    minima = find_peaks(-rdf_from_first_peak[:,1])[0][0]
    inner_cutoff = rdf[extrema + minima]
    if plot:
        fig, ax = plt.subplots()
        data_adr = str(data).split()[-1].strip('>')
        ax.plot(rdf[:,0], rdf[:,1])
        ax.scatter(*inner_cutoff)
        fig.savefig(data_adr)
    print(f'Using inner_cutoff: {inner_cutoff[0]}')
    inner_cutoff = inner_cutoff[0]
    return inner_cutoff

