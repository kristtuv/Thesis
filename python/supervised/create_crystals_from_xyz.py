import ase.spacegroup
import numpy as np
from ase.io import write 
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
import ase.units as units
from ase.calculators.emt import EMT
from ase.io import read
from ovito.io.ase import ase_to_ovito
from ovito.io import export_file, import_file
from ovito.modifiers import ReplicateModifier, WrapPeriodicImagesModifier
import ase.geometry
from os import listdir
from os.path import isfile
from ase.io.extxyz import XYZError
import shutil
from util import is_dir
from os import walk
from os.path import join
import re

def replicate_cell(file_name, dump_name, replicate=None, n_minimum_particles=500):
    print('File Name: ', file_name)
    xyz_data = read(file_name)
    with open(file_name, 'r') as f:
        lines = f.read()
        cell_vec = re.compile('a\(\d\)\s+=\s+(.*)\n')
        cell_vec = cell_vec.findall(lines)
        Lx = list(filter(None, cell_vec[0].strip().split(' ')))
        Lx = np.array(Lx, dtype=np.float64)
        Ly = list(filter(None, cell_vec[1].strip().split(' ')))
        Ly = np.array(Ly, dtype=np.float64)
        Lz = list(filter(None, cell_vec[2].strip().split(' ')))
        Lz = np.array(Lz, dtype=np.float64)
        L = np.stack([Lx, Ly, Lz], axis=1)
        print(L)
        # L[[0, 1, 2], [0, 1, 2]] -= 1e-15
    print(L)
    data = ase_to_ovito(xyz_data)
    data.cell_[:, :3] = L
    data.cell_[:, -1] = 1e-5
    data.cell_.pbc = [True]*3
    data.apply(WrapPeriodicImagesModifier())
    # export_file(data, f'{dump_name}.data', 'lammps/data')
    # exit()
    data = data.to_ase_atoms()
    ase.geometry.get_duplicate_atoms(data, cutoff=0.1, delete=True)
    data = ase_to_ovito(data)
    n_particles = data.number_of_particles
    print('Number of particles in unit cell: ', n_particles)
    if not replicate:
        min_particles_after_replicate = n_minimum_particles
        rep = int(np.ceil((min_particles_after_replicate / n_particles)**(1/3)))
        replicate = {'num_x': rep, 'num_y': rep, 'num_z': rep}
    data.apply(ReplicateModifier(**replicate, adjust_box=True))
    n_particles = data.number_of_particles
    print('Number of particles after replicate', n_particles)
    print('DumpPath: ', dumpname)
    export_file(data, f'{dump_name}.data', 'lammps/data')
    print()


if __name__=='__main__':
    file_dir = 'crystal_files_xyz/'
    error_dir = file_dir + 'error_files/' 
    dump_dir = 'crystal_files_unit/'
    is_dir(file_dir)
    is_dir(error_dir)
    is_dir(dump_dir)
    _, _, file_names = next(walk(file_dir))
    dump_names = [dump_dir+f.strip('.xyz') for f in file_names]
    file_names = [file_dir+f for f in file_names]
    for filename, dumpname in zip(file_names, dump_names):
        try:
            replicate_cell(filename, dumpname)
        except (IndexError, KeyError, XYZError, AttributeError) as e:
            error_path = error_dir + filename.rsplit('/', maxsplit=1)[-1]
            print(e)
            print(f"Moving file to {error_path}")
            shutil.move(filename, error_path)
            continue



