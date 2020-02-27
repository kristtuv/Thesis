import os
from nn import make_file, model_exists, predict, write_data
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import re
from keras.models import model_from_json
from util import is_dir

def walk_path(directory, startswith='history'):
    for root, dirs, files in os.walk(directory):
        if files:
            for name in files:
                if name.startswith(startswith):
                    path = os.path.join(root, name)
                    yield path

class EvaluateModel:
    def __init__(
            self,
            result_directories=[],
            dumpdir='results_evaluated/',
            datapath='/home/kristtuv/Documents/master/src/python/quasi/',
            adjacencypath=('/home/kristtuv/Documents/master/src/python/supervised/'
                'quasicrystal_files_adjacency/cutoff_finder/')):

        self.result_directories = result_directories
        self.dumpdir = dumpdir
        self.datapath = datapath
        self.adjacencypath = adjacencypath

    def set_cutoff(self, finder='cutoff_finder'):
        paths = []
        for directory in self.result_directories:
            for path in walk_path(directory):
                if finder in path or finder==None:
                    paths.append(path)
                else:
                    continue
        self.paths = paths
        return paths

    def split_neighbors(self):
        paths = defaultdict(list) 
        for path in self.paths:
            neighbor_pattern = re.compile('numneighbors(\d\d)')
            neighbors = neighbor_pattern.search(path)
            nneigh = int(neighbors.group(1))
            paths[nneigh].append(path)
        self.paths = paths
        return paths

    def sort_models(self, neighbors=[]):
        sorted_models = {}
        if not neighbors:
            neighbors = sorted(self.paths.keys())
        for neighbor in neighbors:
            best_models = []
            for path in self.paths[neighbor]:
                history = np.load(path, allow_pickle=True).item()
                best_models.append((np.max(history['val_accuracy']), path))
            best_models = sorted(best_models, reverse=True)
            sorted_models[neighbor] = best_models
        self.sorted_models = sorted_models
        return sorted_models

    def plot_models(self, num_best=1, neighbors=[], metric='val_loss', dumpname=''):
        if not neighbors:
            neighbors = sorted(self.paths.keys())

        fig, ax = plt.subplots()
        shortest_metric = 150
        for neighbor in neighbors:
            for _, path in self.sorted_models[neighbor][:num_best]:
                history = np.load(path, allow_pickle=True).item()
                eval_metric = history[metric]
                if len(eval_metric) < shortest_metric:
                    shortest_metric = len(eval_metric)
                ax.plot(eval_metric, label=f'Neighbors: {neighbor}', )
        if 'loss' in metric:
            ylabel = 'Loss'
        if 'accuracy' in metric:
            ylabel = 'Accuracy'
        # ax.set_xlim(-1, shortest_metric + 10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', ncol=2, fancybox=True, shadow=True)
        plt.savefig(f'/home/kristtuv/Documents/master/latex/plots/{dumpname}_{metric}_supervised.png', transparent=True)
        plt.show()

class RunModel(EvaluateModel):
    def import_adjacency( self ):
        path = self.adjacencypath
        adjacency_files = defaultdict(list)
        neigh_pattern = re.compile('nneigh(\d\d\d*)')
        for f in walk_path(path, startswith=''):
            neighbors = neigh_pattern.search(f)
            nneigh = int(neighbors.group(1))
            adjacency_files[nneigh].append(f)
        self.adjacency_files = adjacency_files
        return adjacency_files

    def import_datafiles( self ):
        path = self.datapath
        datafiles = []
        for f in walk_path(path, startswith=''):
            if not f.endswith('.log'):
                datafiles.append(f)
        self.datafiles = datafiles
        return datafiles

    def import_models(self, num_best=1, neighbors=[]):
        if not neighbors:
            neighbors = sorted(self.paths.keys())
        best_models = defaultdict(list)
        dump_paths = defaultdict(list)
        for neighbor in neighbors:
            for _, path in self.sorted_models[neighbor][:num_best]:
                path = path.replace('history.npy', 'model')
                pathdir = re.search('^\w*\/', path).group(0)
                with open(path+'.json', 'r') as json_file:
                    loaded_json_file = json_file.read()
                    model = model_from_json(loaded_json_file)
                    model.load_weights(path+'.h5')
                    model.compile(
                            optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                            )
                best_models[neighbor].append(model)
                dump_path = re.sub('^\w*\/', self.dumpdir+pathdir, path)
                dump_paths[neighbor].append(dump_path.rsplit('/', 1)[0] + '/')
        self.best_models = best_models
        self.dump_paths = dump_paths
        return best_models, dump_paths

    def import_labels(self, path='datasets/cutoff_finder/'):
        labels = {}
        for f in walk_path(path, startswith=''):
            if 'labels' in f:
                neighbor_pattern = re.compile('numneighbors(\d\d)')
                neighbors = neighbor_pattern.search(f)
                nneigh = int(neighbors.group(1))
                labels[nneigh] = f
        self.labels = labels
        return labels

    @staticmethod
    def map_labels(counter, labels=None):
       
        try:
            counter.pop(-1)
        except:
            pass
        c = [i[0] for i in counter.most_common()]
        cc = []
        for i in range(len(counter)):
            cc.append((c[i], len(counter)-i-1))
        cc.append((-1, -1))
        cc = dict(cc)
        if labels:
            labels = dict(zip(labels.values(), labels.keys()))
            labels[-1] = 'disordered' 
            new_labels = {}
            for key, value in cc.items():
                new_labels[value] = labels[key]
            return cc, new_labels
        else:
            return cc

    def map_predictions(self, counter, predictions):
        old_pred = predictions
        mapping = self.map_labels(counter)
        new_predictions = []
        for pred in predictions:
            new_predictions.append(mapping[pred])
        return new_predictions
        

    def print_best(self,neighbors=[], num_best=1):
        if not neighbors:
            neighbors = sorted(self.paths.keys())
        best_models, dump_paths= self.import_models(num_best=num_best, neighbors=neighbors)
        string_conv = 'Neighbors & Filters & Activations & Poolsize & Dropout & Accuracy \\\\\n'
        string_dense = 'Neighbors & Layers & Activations & Dropout & Accuracy \\\\\n'
        for neighbor in neighbors:
            for _, path in self.sorted_models[neighbor][:num_best]:
                if 'conv' in path:
                    accuracy = np.max(np.load(path, allow_pickle=True).item()['val_accuracy'])
                    filters = re.search('filters(.*)_kernelsize', path).group(1)
                    filters = list(map(int, filters.split('|')))
                    activations = re.search('activation(.*)_poolsize', path).group(1)
                    activations = list(map(lambda x: x.capitalize(), activations.split('|')))
                    activations = '['+', '.join(activations)+']'
                    poolsize = re.search('poolsize(.*)_epochs', path).group(1)
                    poolsize = list(map(int, poolsize.split('|')))
                    dropout = float(re.search('dropout(.*)/', path).group(1))
                    string_conv += f'{str(neighbor)} & {filters} & {activations} & {poolsize} & {dropout} & {accuracy.round(4)}\\\\\n'

                elif 'dense' in path:
                    accuracy = np.max(np.load(path, allow_pickle=True).item()['val_accuracy'])
                    activations = re.search('activation(.*)_epochs', path).group(1)
                    activations = list(map(lambda x: x.capitalize(), activations.split('|')))
                    activations = '['+', '.join(activations)+']'
                    units = re.search('units(.*)_inputshape', path).group(1)
                    units = list(map(int, units.split('|')))
                    dropout = float(re.search('dropoutrate(.*)/', path).group(1))
                    string_dense += f'{str(neighbor)} & {units} & {activations} & {dropout} & {accuracy.round(4)}\\\\\n'

        print(string_conv)
        print(string_dense)

    def run_models(self, num_best=1, neighbors=[], tol=0.3, save=False, datatype='quasi'):
        if not neighbors:
            neighbors = sorted(self.paths.keys())
        adjacency_files = self.import_adjacency()
        best_models, dump_paths= self.import_models(num_best=num_best, neighbors=neighbors)
        labels = self.import_labels()
        for neighbor in neighbors:
            crystal_labels = np.load(labels[neighbor], allow_pickle=True).item()
            for best_model, dump_path in zip(best_models[neighbor], dump_paths[neighbor]):
                for adjacency in adjacency_files[neighbor]:
                    is_dir(dump_path)
                    dump_name = dump_path+adjacency.rsplit('/')[-1]
                    data = np.load(adjacency)
                    data = data.reshape(*data.shape, 1)
                    try:
                        prediction = predict(best_model, data, tol=tol)
                    except ValueError as e:
                        data = data.reshape(-1, np.prod(data.shape[1:]))
                        prediction = predict(best_model, data, tol=tol)
                    if datatype=='quasi':
                        crystal_labels['disordered'] = -1
                        if save:
                            np.save(dump_name+'_prediction', prediction)
                            np.save(dump_name+'_clusters', Counter(prediction))
                            np.save(dump_name+'_labels', crystal_labels)

                    if datatype=='methane':
                        if save:
                            counter = Counter(prediction)
                            _, new_labels = self.map_labels(counter, crystal_labels)
                            prediction = self.map_predictions(counter, prediction)
                            counter = Counter(prediction)
                            np.save(dump_name+'_prediction', prediction)
                            np.save(dump_name+'_clusters', counter)
                            np.save(dump_name+'_labels', new_labels)
                            write_data(self.datapath, prediction, dump_name)


if __name__=='__main__':
    methane=False
    quasi=True
    if methane:
        directories = ['results_conv_sweep/', 'results_dense/', 'results_conv_methane/']
        for d in directories:
            rm = RunModel(result_directories=[d],
                    dumpdir='results_evaluated_methane_tol/1560000/',
                    datapath='/home/kristtuv/Documents/master/src/python/Grace_ase/datafiles/restart_step_1560000.data',
                    adjacencypath=('/home/kristtuv/Documents/master/src/python/Grace_ase/new_adjacency/restart_step_1560000/')
                    )
            rm.set_cutoff()
            rm.split_neighbors()
            sorted_models = rm.sort_models()
            rm.run_models(neighbors=[], datatype='methane', save=False)

    if quasi:
        directories = ['results_conv_sweep/', 'results_dense/']
        # directories = ['results_unit_newbreak/']
        for d in directories:
            rm = RunModel(result_directories=[d],
                    dumpdir='results_evaluated_unit_tol03_newbreak/'
                    )
            rm.set_cutoff()
            rm.split_neighbors()
            sorted_models = rm.sort_models()
            # print(sorted_models)
            rm.print_best()
            # rm.run_models(neighbors=[60], datatype='quasi', save=True, tol=0.3)

