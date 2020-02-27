from PyQt5.QtWidgets import QApplication
app = QApplication([])
from operator import itemgetter
from collections import Counter
from ovito.io import import_file
import os
import numpy as np
import freud
from util import is_dir, cutoff_finder, check_frame
from util import get_first_minima, get_first_minima_after_max
import time
import networkx as nx
from networkx import karate_club_graph, to_numpy_matrix, to_scipy_sparse_matrix
import matplotlib.pyplot as plt
from itertools import zip_longest


class Adjacency(object):
    def __init__(self, paths, ):
        self.paths = paths

    def build(self, data, num_neighbors, inner_cutoff, frame, dump_filename, recompute=False):
        if inner_cutoff==None:
            cutoff = cutoff_finder(data)
        elif inner_cutoff=='cutoff_finder':
            cutoff = cutoff_finder(data)
        elif inner_cutoff=='get_first_minima':
            cutoff = get_first_minima(data)
        elif inner_cutoff=='get_first_minima_after_max':
            cutoff = get_first_minima_after_max(data)
        else:
            cutoff=inner_cutoff
        dump_filename = (
                dump_filename
                +f'_frame{frame}_nneigh{num_neighbors}_icut{np.round(cutoff, 3)}.npy'
                )
        print('Computing frame: ', frame)
        if os.path.isfile(dump_filename) and not recompute:
            print(f'Precalculated: Loading {dump_filename}')
            adjacency_matrix = np.load(dump_filename)
            counter = 0
            for m in adjacency_matrix:
                if np.all(m==0):
                    counter += 1
            print('Total empty adjacency matrices: ', counter)
            return adjacency_matrix
        else:
            full_start = time.time()
            N_particles = data.particles.count
            print('Number of particles in data: ', N_particles)
            print('Calculating adjacency matrix')
            sim_cell = data.cell.matrix
            box = freud.box.Box.from_matrix(sim_cell)
            positions = data.particles.positions

            query_args = {'mode': 'nearest', 'num_neighbors': num_neighbors, 'exclude_ii':False}
            aabb = freud.locality.AABBQuery(box, positions)
            nearest_neighbors = aabb.query(positions, query_args=query_args).toNeighborList()
            # point_indices = nearest_neighbors.point_indices
            neighbor_counts = nearest_neighbors.neighbor_counts
            outer_distances = nearest_neighbors.distances.reshape(-1, num_neighbors)
            outer_distances = np.argsort(outer_distances, axis=1)
            # point_indices = nearest_neighbors.point_indices
            outer_list = nearest_neighbors.point_indices.reshape(-1, num_neighbors)
            outer_list = outer_list[np.arange(len(outer_distances)), outer_distances.T].T
            outer_repeat = np.repeat(outer_list, neighbor_counts, axis=0).astype(np.int64)

            nearest_neighbors = nearest_neighbors.filter_r(r_max=cutoff)
            inner_segments = nearest_neighbors.segments
            point_indices = nearest_neighbors.point_indices
            inner_list = np.split(point_indices, inner_segments)[1:]
            inner_list_getter = itemgetter(*outer_list.ravel())(inner_list)

            adjacency_matrix = np.asarray(
                    [np.isin(x, y) for x, y in zip(outer_repeat, inner_list_getter)], dtype=np.int8
                    ).reshape(-1, num_neighbors, num_neighbors)
            adjacency_matrix = adjacency_matrix#*outer_distances
            print('Adjacency calculation time: ', time.time() - full_start)
            counter = 0
            print('Adjacency matrix shape: ', adjacency_matrix.shape)
            for m in adjacency_matrix:
                if np.all(m==0):
                    counter += 1
            print('Total empty adjacency matrices: ', counter)
            print(f'Creating file {dump_filename}')
            np.save(dump_filename, adjacency_matrix)
            print()
            return adjacency_matrix

    def build_adjacency(
            self, coordination,
            frame, dump_filenames, recompute=False,
            num_neighbors=30, inner_cutoff=None
            ):

        paths = self.paths
        adjacency_matrix_list = []
        data_objects = []
        if not inner_cutoff: #not isinstance(inner_cutoff, list):
            inner_cutoff = [inner_cutoff]

        for path, dump_filename, cutoff in zip_longest(paths, dump_filenames, inner_cutoff):
            print()
            print('Cutoff: ', cutoff)
            print('Path: ', path)
            print('Dumpname: ', dump_filename)
            pipe = import_file(path)
            frame = check_frame(frame, pipe)
            pipe.modifiers.append(coordination)
            data = pipe.compute(frame)
            data_objects.append(data)
            print('Filepath indata: ', path)
            adjacency_matrix = self.build(
                    data, num_neighbors, cutoff,
                    frame, dump_filename, recompute=recompute, 
                    )
            adjacency_matrix_list.append(adjacency_matrix)
        
        fulladjacency= np.concatenate(adjacency_matrix_list, axis=0)
        return fulladjacency, data_objects

