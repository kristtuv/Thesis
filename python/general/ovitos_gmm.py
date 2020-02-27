from PyQt5.QtWidgets import QApplication
app = QApplication([])
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
import matplotlib as mpl
import itertools
import sys
from os import listdir
from itertools import product


class Classifier:
    def __init__(self, path, frame=None):
        self.path = path
        pipe = import_file(path)
        if frame < 0:
            frame = pipe.source.num_frames + frame + 1
        data = pipe.source.compute(frame)
        positions = data.particles.positions.array
        sim_cell = data.cell.matrix
        self.box = freud.box.Box.from_matrix(sim_cell)
        particle_type = data.particles.particle_types.array
        self.particle_type = particle_type
        self.data = data
        self.positions = positions

    def remove_particles(self, particle_type=2):
        self.remove_type = particle_type
        self.remaining_positions = self.positions[self.particle_type != self.remove_type] 
        return self.remaining_positions

    def set_lms_values(self, degree, negative_m):
        lms = []
        if negative_m:
            for l in range(degree+1):
                m_range = np.arange(-l, l+1)
                for m in m_range:
                    lms.append((m, l))
            self.lms = np.array(lms)
        else:
            min_m = 0
            for l in range(degree+1):
                m_range = np.arange(min_m, l+1)
                for m in m_range:
                    lms.append((m, l))
            self.lms = np.array(lms)
        return self

    def get_sph(self, positions, n_neighbors=4, l=7, rmax=3, negative_m=False, mode='neighborhood'):
        self.n_neighbors = n_neighbors
        self.l = l
        self.negative_m = negative_m
        self.set_lms_values(l, negative_m)
        box = self.box
        localdescriptors = freud.environment.LocalDescriptors(l, negative_m, mode=mode)
        aabb = freud.locality.AABBQuery(box, positions)
        query_args = {'mode':'nearest', 'num_neighbors': n_neighbors, 'exclude_ii':True}
        sph = localdescriptors.compute(system=aabb, query_points=positions, neighbors=query_args).sph
        self.sph = sph
        return self

    def get_sph_mean(self, mode='local'):
        sph = self.sph
        sph  = sph.reshape(-1, self.n_neighbors, sph.shape[1])
        if mode=='local':
            sph_sum = np.sum(sph, axis=1)
            sph_abs = np.abs(sph_sum)
            sph_mean = sph_abs/self.n_neighbors
            self.sph = sph_mean
        if mode=='global':
            sph_sum = np.sum(np.sum(sph, axis=1), axis=0)
            sph_abs = np.abs(sph_sum)
            sph_mean = sph_abs/(self.n_neighbors*sph.shape[0])
            self.sph = sph_mean
        return self

    def exclude_sph(self, exclude_degrees=[0]):
       degrees = np.arange(0, self.l +1)
       idx = np.arange(self.sph.shape[1])
       if self.negative_m: m_idx = np.cumsum(2*degrees + 1)
       else: m_idx = np.cumsum(degrees + 1)
       #Insert zero at beginning to make indexing easier
       m_idx = np.insert(m_idx, 0, 0)
       #Find all indices to remove
       remove_idx = [a for l in exclude_degrees for a in range(m_idx[l], m_idx[l+1])]
       idx = np.delete(idx, remove_idx)
       self.sph = self.sph[:, idx]
       return self

    def compute_gmm(self, features, cluster=3, all_particles=True):
        clusters = GaussianMixture(cluster, max_iter=500,
                                   tol=1e-5, verbose=True).fit(features)
        clusters = clusters.predict(features)

        if all_particles:
            self.clusters=clusters
            return clusters
        else:
            remaining_particles = np.array(self.particle_type != self.remove_type,
                                           dtype=np.int)
            idx = np.argwhere(remaining_particles).flatten()
            remaining_particles[idx] = clusters
            self.clusters=remaining_particles
            return remaining_particles

    def dump_data(self, filename):
       data = self.data
       data.particles_.create_property('ClusterNumber', data=self.clusters)
       export_file(data, filename, "lammps/dump",
               columns=["Particle Type", 'Position.X', 'Position.Y', 'Position.Z', "ClusterNumber"])

    def mergeClusters(self, gmm, data):
        p = gmm.predict_proba(data)
        mergeMap = np.arange(gmm.n_components)
        components = set(mergeMap)
        bestEntropies = [-np.nansum(p*np.log(p))]
        print(bestEntropies)
        clusterMaps = [mergeMap.copy()]
        mergeCounts = []
        while len(components) > 1:
            entropies = []
            for (k, kprime) in itertools.combinations(components, 2):
                joint = p[:, k] + p[:, kprime]
                S = (-np.nansum(p[:, k]*np.log(p[:, k])) - 
                     np.nansum(p[:, kprime]*np.log(p[:, kprime])) + 
                     np.nansum(joint*np.log(joint)))
                entropies.append((k, kprime, S))
                
            entropies = list(sorted(entropies, key=lambda x: -x[2]))
            (k, kprime, best_S) = entropies[0]
            
            dest, src = sorted([k, kprime])
            p[:, dest] += p[:, src]
            p[:, src] = 0
            mergeMap[mergeMap == src] = dest
            bestEntropies.append(-np.nansum(p*np.log(p)))
            clusterMaps.append(mergeMap.copy())
            mergeCounts.append(np.sum(np.argmax(p, axis=-1) == dest))
            components.remove(src)
        return clusterMaps
        # return bestEntropies[::-1], clusterMaps[::-1], mergeCounts[::-1]

def cond(x):
    #Sort condition
    k = float(x.split('_k')[1].split('_phi')[0])
    p = float(x.split('_phi')[1].split('_T')[0])
    return (k, p)

if __name__=='__main__':
    path = '/home/kristtuv/Downloads/polycrystalline_shear_snapshots/polycrystal_L100.0/continuation_step_1000000_T_263.15/continuation_stress_calculation_max_with_deform_temp/restart_step_1530000.data'
    filedir = 'allquasidata/'
    dumpdir = 'results_quasi/'

    k = np.linspace(5.8, 9.5, 10).round(3)
    phi = np.linspace(0.38, 0.8, 10).round(3)
    potentials = list(product(k, phi) )

    #Import files
    files = listdir(filedir)
    cluster_files = sorted([filedir+f for f in files if not f.endswith('.log') and 'quasi' in f], key=cond)
    ks = [float(f.split('_k')[1].split('_phi')[0]) for f in cluster_files]
    phis = [float(f.split('_phi')[1].split('_T')[0]) for f in cluster_files]

    #Remove files which was not run
    keep_idx = []
    for idx, a in enumerate(zip(ks, phis)):
        if a in potentials:
            keep_idx.append(idx)
    cluster_files = [cluster_files[i] for i in keep_idx]
    ks = [ks[i] for i in keep_idx]
    phis = [phis[i] for i in keep_idx]

    #Create mesh
    # oxygen_positions = model.remove_particles(2)
    # model.get_sph(oxygen_positions, n_neighbors=4, l=7, negative_m=False, mode='global')
    sphs = []
    for f in cluster_files:
        model = Classifier(f, frame=-1)
        model.get_sph(model.positions, n_neighbors=4, l=7, negative_m=False, mode='global')
        model.get_sph_mean(mode='global')
        sph = model.sph
        sphs.append(sph)
    sphs = np.array(np.stack(sphs, axis=0))
    model.compute_gmm(sphs, cluster=10, all_particles=True)
    # print(model.clusters)
    print(*zip(ks, phis, cluster_files, model.clusters), sep='\n')
    # model.dump_data('testingdata/penis.lammps')
    x = np.unique(ks)
    y = np.unique(phis)
    x = np.pad(x, (0, 1), constant_values=(2*x[-1]-x[-2]))
    y = np.pad(y, (0, 1), constant_values=(2*y[-1]-y[-2]))
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(x), len(y)))

    for c, phi, k, cluster in zip(cluster_files, phis, ks, model.clusters):
        #Set Z index

        j = np.where(x == k)
        i = np.where(y == phi)
        Z[i, j] = cluster
    Z[-1],Z[:, -1] = Z[-2], Z[:, -2]
    ticks = np.unique(Z).astype(int)
    bounds = [-1.5] + list(np.unique(Z).astype(int) + 0.5)
    cmap = plt.get_cmap('RdGy', len(bounds)-1)#, len(bounds)+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    cax = ax.pcolor(X, Y, Z, cmap = cmap, edgecolors='w')
    cbar = fig.colorbar(cax, norm=norm, boundaries=bounds, ticks=ticks)
    plt.show()





















    exit()

    n_clusters = np.arange(20, 30)
    models = [model.compute_gmm(model.sph, cluster=i, all_particles=False) for i in n_clusters]
    plt.plot(n_clusters, [m.bic(model.sph) for m in models])
    plt.show()
    exit()
    # model.compute_sph(hydrogen_positions, neighbors=16, l=7, negative_m=True)
    model.compute_sph(hydrogen_positions, neighbors=4, l=7, negative_m=True)
    # c = model.compute_chill()
    model.compute_gmm(c, cluster=3, all_particles=False)
