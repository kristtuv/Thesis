import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from ovito.io import import_file, export_file
from ovito.modifiers import CoordinationAnalysisModifier
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from adjacency import Adjacency
from autoencoder import AutoEncoder
from util import is_dir, check_frame
from collections import Counter
import os
from itertools import product

class Datapreparation:
    def __init__(self, paths:list, dump_params:dict={}):
        params = {'adjacency_dumpdir':'dumpfiles/adjacency/', 'encoded_dumpdir':'dumpfiles/encoded/',
                  'dumpdir':'dumpfiles/', 'dump_filenames':None, 
                  'encoded_filename':'MergedAdjacency', **dump_params, }
        #Set all params as self
        self.dict_to_variables(params)
        if not isinstance(paths, list):
            paths = [paths]
        if self.dump_filenames is None:
            self.dump_filenames = [path.split('/')[-1] for path in paths]
        is_dir(self.dumpdir)
        is_dir(self.encoded_dumpdir)
        is_dir(self.adjacency_dumpdir)

        self.adjacency_filenames = [
                self.adjacency_dumpdir+name+'_adjacency' for name in self.dump_filenames
                ]
        if len(paths) == 1:
            self.encoded_filename = self.encoded_dumpdir+self.dump_filenames[0]+'_encoded'
        else: 
            self.encoded_filename = self.encoded_dumpdir+self.encoded_filename+'_encoded'
        self.paths = paths


    def compute_adjacency(self, frame, adjacency_params={},):
        params = {'coordination_cutoff':10, 'number_of_bins':100,
                  'num_neighbors':30, 'inner_cutoff':None,
                  'recompute_adjacency':False, **adjacency_params}
        self.dict_to_variables(params)
        coordination = CoordinationAnalysisModifier(
                cutoff=self.coordination_cutoff,
                number_of_bins=self.number_of_bins
                )
        adj = Adjacency(self.paths)
        self.data, data_objects = adj.build_adjacency(
                coordination = coordination,
                frame = frame,
                dump_filenames = self.adjacency_filenames,
                num_neighbors = self.num_neighbors,
                inner_cutoff = self.inner_cutoff,
                recompute = self.recompute_adjacency
                )
        return self.data.astype(np.int8)

    def compute_autoencoder(self, frame, data=None, autoencoder_params={}):
        params = {'epoch':2, 'autoconstruct_power':4,
                  'recompute_autoencoder':False,
                  'input_layers':None, 'activations':None,
                  'model_name': 'test', **autoencoder_params, }
        if data:
            self.data=data
        self.dict_to_variables(params)
        ae = AutoEncoder(self.epoch, frame, self.encoded_filename+f'_neighbors{self.num_neighbors}')
        ae.train_test(self.data)
        self.data, encoded_filename = ae.autoencoder(
                input_layers=self.input_layers,
                activations=self.activations,
                recompute=self.recompute_autoencoder,
                power_start = self.autoconstruct_power,
                )
        return self.data, encoded_filename

    def compute_pca(self, data=None):
        if data is None:
            data = self.data

        if data.ndim > 2:
            print(f'Data shape is {data.ndim}, raveling down to 2 dims')
            data = data.reshape(len(data), np.prod(data.shape[1:]))
        pca = PCA().fit(data)
        explained_v = pca.explained_variance_ratio_
        cum = np.cumsum(explained_v)
        idx = np.argmax(cum > 0.9)
        return pca.transform(data)[:, :idx]

    def dict_to_variables(self, args_dict):
        for key, value in args_dict.items():
            setattr(self, key, value)

class Cluster:
    def __init__(self, data):
        self.scaler = StandardScaler()
        self.data = data

    def evaluate(self, data, labels):
        calinski = metrics.calinski_harabasz_score(data, labels)
        print('Calinski, high good: ', calinski)
        silhouette = metrics.silhouette_score(data, labels)
        print('Silhouette, high good: ', silhouette)
        davies = metrics.davies_bouldin_score(data, labels)
        print('Davies, low good: ', davies)
        return {'calinski': calinski, 'silhouette': silhouette, 'davies': davies}
        
    def agglomerative(self, params):
        for c in params:
            print(f"Running agglomerative_model")
            agglomerative_model = AgglomerativeClustering(n_clusters=c).fit(self.data)
            labels = agglomerative_model.labels_
            try:
                evaluate = (self.evaluate(self.data, labels), Counter(labels))
            except ValueError:
                evaluate = Counter(labels)
            yield labels, evaluate
        print('Finished agglomerative')

    def gauss(self, params):
        for c in params:
            gauss_model = GaussianMixture(c).fit(self.data)
            print(f"Running gauss_model")
            labels = gauss_model.predict(self.data)
            try:
                evaluate = (self.evaluate(self.data, labels), Counter(labels))
            except ValueError:
                evaluate = Counter(labels)
            yield labels, evaluate
        print('Finished gauss')

    def optics(self, params):
        for eps, sample in params:
            print(f"Running optics_model")
            optics_model = OPTICS(eps=eps,min_samples=sample).fit(self.data)
            labels = optics_model.labels_
            try:
                evaluate = (self.evaluate(self.data, labels), Counter(labels))
            except ValueError:
                evaluate = Counter(labels)
            yield labels, evaluate
        print('Finished optics')

    def dbscan(self, params):
        for eps, sample in params:
            print(f"Running dbscan_model")
            dbscan_model = DBSCAN(eps=eps, min_samples=sample).fit(self.data)
            labels = dbscan_model.labels_
            try:
                evaluate = (self.evaluate(self.data, labels), Counter(labels))
            except ValueError:
                evaluate = Counter(labels)
            yield labels, evaluate
        print('Finished dbscan')


    def save_ovito(self, cluster_data, path, frame=-1, dumpname=None, dumpdir='', ext='optics'):
        pipe = import_file(path)
        frame = check_frame(frame, pipe)
        if dumpname is None:
            dumpname = path.split('/')[-1] + f'_frame{frame}_cluster_{ext}'
        data = pipe.compute(frame)
        data.particles_.create_property('Cluster', data=cluster_data)
        dumpname = dumpdir+dumpname
        print(dumpname)
        print(data.particles_.properties)
        
        
        export_file(data, dumpname, 'lammps/dump',
                columns=['Particle Type', 'Position.X', 'Position.Y', 'Position.Z', 'Cluster']
                )

def argparser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--frame', default=-1, type=int)
    parser.add_argument('--outer_shell', default=2, type=int)
    parser.add_argument('--inner_shell', default=1, type=int)
    parser.add_argument('--epoch', default=30, type=int)
    return parser.parse_args()

def get_triu(matrix):
    triu = np.triu_indices(matrix.shape[1])
    return matrix[:, triu[0], triu[1]]

def check_paths(dump, clusters=None, epsilon=None, min_samples=None):
    if clusters:
        data_return = clusters.copy()
        dumpnames = []
        for c in clusters:
            dumpname = dump.format(f'cluster{c}')
            if os.path.isfile(dumpname):
                data_return.remove(c)
            else:
                dumpnames.append(dumpname)

    elif epsilon and min_samples:
        combinations = list(product(epsilon, min_samples))
        data_return = combinations.copy()
        dumpnames = []
        for combination in combinations: 
            eps, sample = combination
            dumpname = dump.format(f'sample{sample}_eps{eps}')
            if os.path.isfile(dumpname):
                data_return.remove(combination)
            else:
                dumpnames.append(dumpname)
    return data_return, iter(dumpnames)



if __name__=='__main__':

    args = argparser()
    frame = args.frame
    outer_shell = args.outer_shell
    inner_shell = args.inner_shell
    epoch = args.epoch
    # datapath = '../allquasidata/'
    # paths = listdir(datapath)
    # paths = [p for p in paths if not p.endswith('.log') ]
    # full_paths = [datapath+p for p in paths]
    # full_paths = ['datafiles/waterpositions320.dump', 'datafiles/restart_step_1530000.data']
    full_paths = ['datafiles/restart_step_1530000.data']
    frame = 0
    # neighbors = [20, 30, 40, 50, 60, 70]
    neighbors = [50, 60]
    # datatypes = ['pca', 'autoencoder']
    datatypes = ['autoencoder']

    eps_optics = [0.0001, 0.001, 0.01, 0.1, 1, 10, 20]
    # eps_optics = [1, 10, 20]
    min_samples_optics = [2, 5, 10, 20, 40, 50]
    eps_dbscan = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    min_samples_dbscan = [2, 5, 10, 20, 40, 100, 200]
    n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for datatype in datatypes:
        for full_path in full_paths:
            for neighbor in neighbors:
                adjacency_params = {'num_neighbors': neighbor, 'inner_cutoff': None, 'recompute_adjacency':False}
                dataprep = Datapreparation(full_path)
                adj = dataprep.compute_adjacency(frame, adjacency_params=adjacency_params)
                if datatype == 'pca':
                    dump_agglomerative = 'hydrate/agglomerative/'+full_path.split('/')[-1] + '_pca_{}_'+f'nneigh{neighbor}'
                    dump_gauss = 'hydrate/gauss/'+full_path.split('/')[-1] + '_pca_{}_'+f'nneigh{neighbor}'
                    dump_optics = 'hydrate/optics/'+full_path.split('/')[-1] + '_pca_{}_'+f'nneigh{neighbor}'
                    dump_dbscan = 'hydrate/dbscan/'+full_path.split('/')[-1] + '_pca_{}_'+f'nneigh{neighbor}'
                    params_agglomerative, dumpname_agglomerative = check_paths(dump_agglomerative, clusters=n_clusters) 
                    params_gauss, dumpname_gauss = check_paths(dump_gauss, clusters=n_clusters) 
                    params_optics, dumpname_optics = check_paths(dump_optics, epsilon=eps_optics, min_samples=min_samples_optics)
                    params_dbscan, dumpname_dbscan = check_paths(dump_dbscan, epsilon=eps_dbscan, min_samples=min_samples_dbscan)
                    print(params_agglomerative)
                    print(params_gauss)
                    print(params_optics)
                    print(params_dbscan)


                    if params_agglomerative or params_gauss or params_optics or params_dbscan:
                        data = dataprep.compute_pca(adj.reshape(-1, neighbor*neighbor))
                    else:
                        continue

                    cluster = Cluster(data)
                    for params in cluster.agglomerative(params_agglomerative):
                        dumpname = next(dumpname_agglomerative)
                        cluster.save_ovito(params[0], full_path, frame=frame, dumpname=dumpname)
                        np.save(dumpname+'_evaluation', params[1])

                    for params in cluster.gauss(params_gauss):
                        dumpname = next(dumpname_gauss)
                        cluster.save_ovito(params[0], full_path, frame=frame, dumpname=dumpname)
                        np.save(dumpname+'_evaluation', params[1])

                    for params in cluster.optics(params_optics):
                        dumpname = next(dumpname_optics)
                        cluster.save_ovito(params[0], full_path, frame=frame, dumpname=dumpname)
                        np.save(dumpname+'_evaluation', params[1])

                    for params in cluster.dbscan(params_dbscan):
                        dumpname = next(dumpname_dbscan)
                        cluster.save_ovito(params[0], full_path, frame=frame, dumpname=dumpname)
                        np.save(dumpname+'_evaluation', params[1])

                else:
                    epoch = 100
                    # layers_list = [None, [1500, 1000], [1500, 1000, 500], [1500, 1000, 500, 256]]
                    autoconstruct = [3, 4, 5, 6, 7, 8]
                    layers_list = [None]
                    activations=[['relu']]
                    for layers in layers_list:
                        for activation in activations:
                            for auto in autoconstruct:
                                if layers is not None:
                                    activation = activation*len(layers)
                                autoencoder_params = {
                                        'epoch': epoch, 'recompute_autoencoder':False, 'autoconstruct_power':auto,
                                        'input_layers': layers, 'activations': activation
                                        }
                                data, dumpname_encoder = dataprep.compute_autoencoder(frame, autoencoder_params=autoencoder_params)
                                current_encoder = dumpname_encoder.split('/')[-1].replace('.npy', '')
                                dump_agglomerative = 'hydrate/agglomerative/'+current_encoder+'_{}'
                                dump_gauss = 'hydrate/gauss/'+current_encoder+'_{}'
                                dump_optics = 'hydrate/optics/'+current_encoder+'_{}'
                                dump_dbscan = 'hydrate/dbscan/'+current_encoder+'_{}'
                                params_agglomerative, dumpname_agglomerative = check_paths(dump_agglomerative, clusters=n_clusters) 
                                params_gauss, dumpname_gauss = check_paths(dump_gauss, clusters=n_clusters) 
                                params_optics, dumpname_optics = check_paths(dump_optics, epsilon=eps_optics, min_samples=min_samples_optics)
                                params_dbscan, dumpname_dbscan = check_paths(dump_dbscan, epsilon=eps_dbscan, min_samples=min_samples_dbscan)
                                print(params_agglomerative)
                                print(params_gauss)
                                print(params_optics)
                                print(params_dbscan)

                                cluster = Cluster(data)
                                for params in cluster.agglomerative(params_agglomerative):
                                    dumpname = next(dumpname_agglomerative)
                                    cluster.save_ovito(params[0], full_path, frame=frame, dumpname=dumpname)
                                    np.save(dumpname+'_evaluation', params[1])

                                for params in cluster.gauss(params_gauss):
                                    dumpname = next(dumpname_gauss)
                                    cluster.save_ovito(params[0], full_path, frame=frame, dumpname=dumpname)
                                    np.save(dumpname+'_evaluation', params[1])

                                # for params in cluster.optics(params_optics):
                                #     dumpname = next(dumpname_optics)
                                #     cluster.save_ovito(params[0], full_path, frame=frame, dumpname=dumpname)
                                #     np.save(dumpname+'_evaluation', params[1])

                                # for params in cluster.dbscan(params_dbscan):
                                #     dumpname = next(dumpname_dbscan)
                                #     cluster.save_ovito(params[0], full_path, frame=frame, dumpname=dumpname)
                                #     np.save(dumpname+'_evaluation', params[1])
