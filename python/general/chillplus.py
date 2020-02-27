from ovito.io import import_file
from ovito.data import NearestNeighborFinder
import numpy as np
from scipy.special import sph_harm
from collections import defaultdict

class ChillPlus:
    def __init__(self, l, nn, filename='../lammps/hydrate/waterpositions10.dump'):
        self.used_frames = []
        self.correlations_allframes = []
        self.correlation_dict = dict()
        self.__pipeline = import_file(filename, multiple_frames=True)
        self.n_frames = self.__pipeline.source.num_frames
        self.l = l
        self.nn = nn

    def __call__(self, frame):
        if frame in self.used_frames:
            print(f'Frame {frame} as already run')
            pass
        else:
            self.used_frames.append(frame)
            data = self.__pipeline.source.compute(frame)
            nearest_neighbors, local_order = self.neighbors(self.nn, data)
            c = self.correlation(nearest_neighbors, local_order)
            self.correlation_dict[frame] = c


    @staticmethod
    def spherical_harmonics(l, angles):
        return np.array([sph_harm(m, l, *angles) for m in range(-l, l+1)])

    @staticmethod
    def cartesian_to_spherical(positions):
        x, y, z = np.float64(positions)
        r = np.linalg.norm(positions)
        theta = np.arctan2(y, x, dtype=np.float64) + np.pi #Azimuthal
        phi = np.arccos(z/r, dtype=np.float64) #Inclination
        return theta, phi

    @staticmethod
    def count_eclipsed(correlation_list):
        return sum(-0.34<e<0.25 for e in correlation_list)

    def neighbors(self, nn, data):
        """
        NearestNeighborFinder uses as index the order of atoms in
        the dump. Because of this, the dump should be sorted by atom
        id. NearestNeighborFinder uses an index which starts at zero
        but the dump id order starts at one. This is remedied by
        adding +1 to the key index of the neighbor_dict
        """
        nn_finder = NearestNeighborFinder(nn, data)
        identifiers = data.particles.identifiers.array #Identifiers from dump
        local_order = dict()
        nearest_neighbors = defaultdict(list)

        for idx, identifier in enumerate(identifiers):
            qlm = complex()
            for neigh in nn_finder.find(idx):
                neighbor_id = identifiers[neigh.index]
                nearest_neighbors[identifier].append(neighbor_id)
                positions = neigh.delta
                angles = self.cartesian_to_spherical(positions)
                qlm += self.spherical_harmonics(self.l, angles)
            qlm = qlm/float(self.nn)

            local_order[identifier] = qlm
        return nearest_neighbors, local_order


    def correlation(self, nearest_neighbors, local_order):
        chill_dict = {}
        for atom, neighbor_list in nearest_neighbors.items():
            atom_i = local_order[atom]
            atom_i_len = np.linalg.norm(atom_i)
            neighbor_correlations = []
            for nn in neighbor_list:
                atom_j = local_order[nn]
                atom_j_len = np.linalg.norm(atom_j)
                c = np.real(np.vdot(atom_j, atom_i)/(atom_j_len*atom_i_len))
                neighbor_correlations.append(c)
            self.correlations_allframes.extend(neighbor_correlations)
            eclipsed_neighbors = self.count_eclipsed(neighbor_correlations)
            chill_dict[atom] = eclipsed_neighbors
        return chill_dict

    def n_bonds(self, frame):
        d = self.correlation_dict[frame]
        four = sum(i==4 for i in d.values())
        three = sum(i==3 for i in d.values())
        two = sum(i==2 for i in d.values())
        one = sum(i==1 for i in d.values())
        return one, two, three, four


if __name__ == '__main__':
    import os
    hydrate = '../lammps/hydrate/'
    # filename = hydrate+'waterpositions250.bin'
    f = 'cubic_ice250'
    f = 'cubic_ice400'
    f = 'hex_ice250'
    filename = hydrate+f+'.dump'
    x = ChillPlus(3, 4, filename=filename)
    x(101)
    # print(x.n_bonds(301))
    # print(x.correlations_allframes)
    # for frame in range(100, 101):
    #     x(frame)
        # print(x.n_bonds(frame))
    np.savetxt(f+'.txt', x.correlations_allframes)


    exit()
    files = sorted([f for f in os.listdir(hydrate) if 'water' in f])
    for f in files:
        file = hydrate + f
        print(f)
        x = ChillPlus(3, 4, filename=file)
        for i in range(40, 50):
            x(i)
            print(x.n_bonds(i))
        np.savetxt(f'{f}.txt', x.correlations_allframes)

