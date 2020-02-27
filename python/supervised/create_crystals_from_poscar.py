import numpy as np
from ovito.io import export_file, import_file
from ovito.modifiers import ReplicateModifier
from util import is_dir
from os import walk

def replicate_cell(file_name, dump_name, replicate=None, n_minimum_particles=500):
    print('File Name: ', file_name)
    pipe = import_file(file_name)
    data = pipe.compute()
    n_particles = data.number_of_particles
    if not replicate:
        min_particles_after_replicate = n_minimum_particles
        rep = int(np.ceil((min_particles_after_replicate / n_particles)**(1/3)))
        replicate = {'num_x': rep, 'num_y': rep, 'num_z': rep}
    data.apply(ReplicateModifier(**replicate, adjust_box=True))
    new_n_particles = data.number_of_particles
    unique = len(np.unique(data.particles.positions[...], axis=0))
    if unique != new_n_particles:
        print(new_n_particles, unique)
        print(file_name)
    export_file(data, f'{dump_name}.data', 'lammps/data')

if __name__=='__main__':
    file_dir = 'crystal_files_poscar/'
    error_dir = file_dir + 'error_files/' 
    dump_dir = 'crystal_files_unit_poscar/'
    is_dir(error_dir)
    is_dir(dump_dir)
    _, _, file_names = next(walk(file_dir))
    dump_names = [dump_dir+f.strip('.xyz') for f in file_names]
    file_names = [file_dir+f for f in file_names]
    for filename, dumpname in zip(file_names, dump_names):
        replicate_cell(filename, dumpname)
