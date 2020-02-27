from ovito.io import import_file, export_file
from ovito.modifiers import AssignColorModifier
from ovito.modifiers import ColorCodingModifier
import numpy as np
import freud
import sklearn
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import cm
import numpy.ma as ma
import matplotlib.pyplot as plt
from ovito.data import CutoffNeighborFinder, NearestNeighborFinder

class NGA:
    def __init__(self, path, frame=None):
        pipe = import_file(path)
        data = pipe.source.compute(frame)
        positions = data.particles.positions.array
        finder = NearestNeighborFinder(12, data)
        finder = CutoffNeighborFinder(5.5, data)
        particle_type = data.particles.particle_type.array
        for i in range(data.particles.count):
            print(particle_type[i])
            neigh = finder.find(i)
            n = []
            n.append(i)
            for neigh in finder.find(i):
                n.append(neigh.index)
            print(n)
            if i==5:
                break

        sim_cell = data.cell.matrix
        cell =  (sim_cell[0,0], sim_cell[1,1], sim_cell[2,2], sim_cell[1,0],
                 sim_cell[0,2], sim_cell[1,2])
        print(cell)
        particle_type = data.particles.particle_types.array
        self.particle_type = particle_type
        self.data = data
        self.box = freud.box.Box(*cell)
        self.positions = positions

    def remove_particles(self, particle_type=2):
        self.remove_type = particle_type
        self.remaining_positions = self.positions[self.particle_type != 2] 
        return self.remaining_positions

    def get_neighbors(self, positions=None, n_neighbors=4, rmax=3):
        if positions==None:
            positions=self.positions
        box = self.box
        nn = freud.locality.NearestNeighbors(rmax, n_neighbors)
        nn_list = nn.compute(box, positions, positions).nlist
        print(np.array(nn_list).reshape(-1, n_neighbors, 2))
        print(nn_list)
        return self


if __name__=='__main__':
    path = '/home/kristtuv/Downloads/polycrystalline_shear_snapshots/polycrystal_L100.0/continuation_step_1000000_T_263.15/continuation_stress_calculation_max_with_deform_temp/restart_step_1530000.data'
    NGA(path).get_neighbors()
