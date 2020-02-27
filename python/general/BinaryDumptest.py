from BinaryDump import BinaryDump
import freud
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.mixture import GaussianMixture

def readlammpsdata(path):
    with open(path, 'r') as f:
        f.readline()
        natoms = int(f.readline().split()[0])
        print(natoms)
        f.readline()
        xline = f.readline().split()
        yline = f.readline().split()
        zline = f.readline().split()
        skew = f.readline().split()
        f.readline()
        f.readline()
        f.readline()

        box =  {'xlo': float(xline[0]), 'xhi': float(xline[1]), 
                'ylo': float(yline[0]), 'yhi': float(yline[1]), 
                'zlo': float(zline[0]), 'zhi': float(zline[1]), 
                'xy': float(skew[0]), 'xz': float(skew[1]), 'yz': float(skew[2])}
        positions = np.loadtxt(f, max_rows=natoms)[:, 3:]
        return box , positions


class AtomComputations(object):
    def __init__(self, filename):
        path = '/home/kristtuv/Downloads/polycrystalline_shear_snapshots/polycrystal_L100.0/continuation_step_1000000_T_263.15/continuation_stress_calculation_max_with_deform_temp/restart_step_1530000.data'
        box, self.positions = readlammpsdata(path)
        # binary_dump = BinaryDump('../lammps/hydrate/waterpositions320.bin')
        # header, atoms = binary_dump.getNextTimestep()

        Lx = box['xhi'] - box['xlo']
        Ly = box['yhi'] - box['ylo']
        Lz = box['zhi'] - box['zlo']
        xy = box['xy']
        xz = box['xz']
        yz = box['yz']
        # atom_id = atoms[:, 0].astype(np.int32)
        # atom_id_dict = {i:atom_id[i] for i in range(len(atom_id))}

        self.box = freud.box.Box(Lx, Ly, Lz, xy, xz, yz)
        # self.positions = atoms[:, 2:]

        nn = self.computeNN(4, 3)
        nl = nn.nlist
        compute = freud.environment.LocalDescriptors(4, 7, 3)
        compute.compute(self.box, 4, self.positions, nlist=nl)
        sph = compute.sph
        print(sph.shape)
        sph = sph[:, 1:]
        sph = np.vsplit(sph, sph.shape[0]/4)
        sph_abs = np.abs(sph)
        sph_mean = np.mean(sph_abs, axis=1)
        
        # print(dir(sklearn))
        gmm = GaussianMixture(10, verbose=True)
        pred = np.array(gmm.fit_predict(sph_mean))
        np.set_printoptions(linewidth=np.infty, threshold=6000)
        print(type(pred))
        print(len(pred))
        # hist = np.histogram(pred)
        # print(hist)
        # gmm.fit(sph_mean)

        # a = gmm.score(sph_mean)
        # a.score(sph)
        # print(a)

        # print(gmm.means_)
        # print(gmm.n_iter_)
        # print(sph_mean)
        # compute.delete[:, 0]
        # print(gmm.converged_)
        # print(compute.sph.shape)



        # self.getSphericalHarmonics(nn, 3)



        # print(atom_id_dict)
        # print(nl.find_first_index(3))
        # print(dir(nl))
        # print(nl.index_i)
        # print(nl.index_j)
        # print(nl.segments)
        # print(nl.neighbor_counts)
        # print(nn.getNeighborList())



        # print(ld.num_particles)
        # print(ld.num_neighbors)
        # print(ld.l_max)
        # print(ld.sph.shape)
        # print(np.sort(ld.sph[0][-7:]))
        # print(np.sort(ld.sph[1][-7:]))
        # print(np.sort(ld.sph[2][-7:]))
        # print(np.sort(ld.sph[3][-7:]))

    @staticmethod
    def index_map(a, b):
        pass


    def Ql(self):
        pass

    def Wl(self):
        pass

    def computeNN(self, n_neighbors=4, r_max=3):
        """
        Number of neighbors
        Guess for start distance to find neighbors
        positions of atoms
        freud box
        """
        box = self.box
        positions = self.positions
        nn = freud.locality.NearestNeighbors(r_max, n_neighbors)
        nn.compute(box, positions)
        return nn

    def getSphericalHarmonics(self, nn, l):
        ld = freud.environment.LocalDescriptors(nn.num_neighbors, l, nn.r_max)
        ld.compute(self.box, nn.num_neighbors, self.positions, nlist=nn.nlist)
        print(ld.sph[0])

    


if __name__=='__main__':
    AtomComputations('hei')



# plt.show()





