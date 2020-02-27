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
        os.makedirs(path)

def get_first_minima(data, plot=False):
    rdf = data.series['coordination-rdf'].as_table()
    # max_height = np.argmax(rdf[:, 1])
    minima = find_peaks(-rdf[:,1])[0]
    inner_cutoff = rdf[minima[0]]
    if plot:
        fig, ax = plt.subplots()
        data_adr = str(data).split()[-1].strip('>')
        ax.plot(rdf[:,0], rdf[:,1])
        ax.scatter(*inner_cutoff)
        fig.savefig(data_adr)
    print(f'Using inner_cutoff: {inner_cutoff[0]}')
    inner_cutoff = inner_cutoff[0]
    return inner_cutoff

#Find from highest
def get_first_minima_after_max(data, plot=False):
    rdf = data.series['coordination-rdf'].as_table()
    max_height = np.argmax(rdf[:, 1])
    minima = find_peaks(-rdf[max_height:,1])[0] + max_height
    inner_cutoff = rdf[minima[0]]
    if plot:
        fig, ax = plt.subplots()
        data_adr = str(data).split()[-1].strip('>')
        ax.plot(rdf[:,0], rdf[:,1])
        ax.scatter(*inner_cutoff)
        fig.savefig(data_adr)
    print(f'Using inner_cutoff: {inner_cutoff[0]}')
    inner_cutoff = inner_cutoff[0]
    return inner_cutoff

#First use
def cutoff_finder(data, plot=False):
    rdf = data.series['coordination-rdf'].as_table()
    max_height = np.max(rdf[:, 1])
    print(max_height)
    peaks = find_peaks(rdf[:, 1], height=max_height*0.4)
    print(peaks)
    extrema = peaks[0][0]
    print(extrema)
    rdf_from_first_peak = rdf[extrema:]
    minima = find_peaks(-rdf_from_first_peak[:,1])[0][0]
    print(minima)
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

