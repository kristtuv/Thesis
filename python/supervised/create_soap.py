from ovito.io import import_file
from create_pearson_from_unit import map_name_to_pearson
import ase
import os
from util import is_dir, cutoff_finder, get_first_minima, get_first_minima_after_max
from ovito.modifiers import CoordinationAnalysisModifier
import numpy as np
import re
from os import listdir
from collections import OrderedDict
from dscribe.descriptors import SOAP
from ovito.io.ase import ovito_to_ase, ase_to_ovito

def sort_key(text):
    text = text.split('.')[0]
    name, *number = text.split('_step')
    return name, int(*number)

def import_files(
        filedir, dumpdir, step_start=None, step_end=None, skip=1,
        n_random=0, keep=[], return_repeatlist=False):
    #Import files
    files = listdir(filedir)
    files.sort(key=sort_key)
    keep_files = []
    if keep:
        for k in keep:
            k1 = k+'_'
            k2 = k+'.'
            keep_files.extend([filedir+f for f in files if k1 in f or k2 in f])
    else:
        keep_files = [filedir+f for f in files]
    files = keep_files
    #Sort files
    #Reme step and return only names
    no_step = [f.split('_step')[0] for f in files]
    #Get index of new file
    unique_struktures, step_index = np.unique(no_step, return_index=True)
    #Devide list into list of stepfiles by name
    split_by_step = np.split(files, step_index)[1:]


    dumpfiles = []
    return_files = []
    repeatlist = []
    for label, f in enumerate(split_by_step):
        if n_random:
            random_files  = np.random.choice(f, n_random, replace=False)
            labels = [label]*n_random
            return_files.extend(list(zip(random_files, labels)))
        else:
            start = int(step_start or 0)
            end = int(step_end or len(f))
            step_files = f[start:end:skip]
            end = int(step_end or len(step_files))
            labels = [label]*(end - start)
            repeatlist.append(len(labels))
            return_files.extend(list(zip(step_files, labels)))

    dumpfiles = [dumpdir+f[0].split('/')[-1] for f in return_files]
    if return_repeatlist:
        return OrderedDict(return_files), dumpfiles, repeatlist
    else:
        return OrderedDict(return_files), dumpfiles


def create_labels(unit_dict):
    unit_dict = {k.split('/')[-1].rsplit('.', maxsplit=1)[0]: v for k, v in unit_dict.items()}
    return unit_dict

def create_dataset(adjacency, ovito_data, cluster_values):
    shape = adjacency.shape
    adjacency = adjacency.reshape(-1, np.prod(shape[1:]))
    indices = [data.number_of_particles for data in ovito_data]
    labels = np.repeat(cluster_values, indices).reshape(-1, 1)
    dataset = np.concatenate((adjacency, labels), axis=1)
    return dataset

def find_cutoffs(files, repeat, coordination, cutoff='cutoff_finder'):
    if cutoff=='cutoff_finder':
        finder = cutoff_finder
    elif cutoff=='get_first_minima':
        finder = get_first_minima
    elif cutoff=='get_first_minima_after_max':
        finder = get_first_minima_after_max
    else:
        return [None]

    cutoffs = []
    for f in files:
        d = import_file(f).compute()
        d.apply(coordination)
        cutoff = finder(d)
        cutoffs.append(cutoff)
    return list(np.repeat(cutoffs, repeat))


def stack_uneven(arrays, fill_value=0.):
    sizes_0 = [a.shape[0] for a in arrays]
    sizes_1 = [a.shape[1] for a in arrays]
    max_1 = np.max(sizes_1)
    result = np.zeros((np.sum(sizes_0), max_1))
    start = 0
    for arr in arrays:
        end = arr.shape[0]
        result[start:start+end, :arr.shape[1]] = arr
        start += end
    return result

def file_handling(filedir, dumpdir):
    is_dir(dumpdir)
    files = [f for f in listdir(filedir) if not f.endswith('.log')]

    dumpfiles = [dumpdir + f for f in files]
    files = [filedir+f for f in files]
    return files, dumpfiles

def get_soap(files, frame=-1, dump_names=[]):
    number_of_bins = 100
    coordination_cutoff = 10
    coordination = CoordinationAnalysisModifier(
            cutoff=coordination_cutoff,
            number_of_bins=number_of_bins
            )
    s = []
    data = []
    for file_name, dump_name in zip(files, dump_names):
        pipe = import_file(file_name)
        if frame < 0:
            frame = pipe.source.num_frames + frame +1
        d = pipe.compute(frame)
        data.append(d)
        d.apply(coordination)
        cutoff = get_first_minima_after_max(d)
        ase_data = ovito_to_ase(d)
        ase_data.symbols = 'X'
        species = np.unique(list(ase_data.symbols))
        print(file_name)
        soap = SOAP(
                rcut=cutoff, 
                periodic=True,
                species=species,
                lmax=7,
                nmax=4,)
        soaps = soap.create(ase_data, ase_data.get_positions())
        s.append(soaps)
        np.save(dump_name, soaps)
        print(soaps.shape)
    return s, data

if __name__=='__main__':
    filedir = '/home/kristtuv/Documents/master/src/python/allquasidata/'
    dumpdir = 'quasicrystal_files_soap/'
    files, dumpfiles = file_handling(filedir, dumpdir)
    soaps, data = get_soap(files)
    for dumpfile, soap in zip(dumpfiles, soaps):
        np.save(dumpfile, soap)

    np.random.seed(1337)
    # Filedirs
    filedir_unit='crystal_files_unit_poscar/'
    filedir_harmonic='crystal_files_harmonic/'
    n_random=None #Do not use#TODO: Fix labels
    step_start=None
    step_end=101
    skip=10
    # cutoff_finders = ['get_first_minima', 'cutoff_finder', 'get_first_minima_after_max', 'None']
    cutoff_finders = ['get_first_minima_after_max']
    for cut in cutoff_finders:
        is_dir(f'datasets_soap/{cut}')
        dataset_dumpname = f'datasets_soap/{cut}/test_soap'
        labels_dumpname = f'datasets_soap/{cut}/test_soap_labels'
        pearson = np.load('utility/pearson_symbols.npy', allow_pickle=True).item()
        pearson['methane_hydrate_I'] = 'ClaI'
        pearson['methane_hydrate_II'] = 'ClaII'
        pearson['methane_hydrate_H'] = 'ClaIV'
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
        dumpdir = 'crystal_files_soap/'
        files = map_name_to_pearson(filedir_unit, pearson, keep=keep, n_keep=200)
        files = [f.replace('.data', '') for f in files]
        files_unit, dumpfiles_unit, repeatlist_unit = import_files(filedir_unit, dumpdir, keep=files, return_repeatlist=True)
        files_harmonic, dumpfiles_harmonic, repeatlist_harmonic = import_files(
                filedir_harmonic,
                dumpdir,
                step_start=step_start,
                step_end=step_end,
                n_random=n_random,
                keep=files,
                skip=skip,
                return_repeatlist=True
                )

        all_files = OrderedDict(**files_unit, **files_harmonic)
        all_files_keys = list(all_files.keys())
        cluster_labels = create_labels(files_unit) #Set of cluster values per strukture
        np.save(labels_dumpname, cluster_labels)
        cluster_values = list(all_files.values()) #Cluster values for every dataset including temperate
        all_dump_files = dumpfiles_unit + dumpfiles_harmonic
        s, data = get_soap(all_files_keys, dump_names=all_dump_files)
        s = np.concatenate(s)
        #Save dataset
        dataset = create_dataset(s, data, cluster_values)
        np.save(dataset_dumpname, dataset.astype(np.int8))
