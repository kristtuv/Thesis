import ase.spacegroup
from ase.io import write 
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
import ase.units as units
from ase.calculators.emt import EMT
from ase import Atoms
from os import listdir
from ovito.io import import_file, export_file
from ovito.io.ase import ase_to_ovito
from ase.calculators.lj import LennardJones
import numpy as np

def run_md_disordered(file_name=None, dump_name=None, steps=None, crystal=None):
    if crystal==None:
        crystal = import_file(file_name).compute().to_ase_atoms()
    calc = LennardJones()
    crystal.set_calculator(calc)
    start_pos = crystal.get_positions()
    dyn = Langevin(crystal, 2*units.fs, 300*units.fs, 0.002)
    prev_step=0
    counter = 0
    for step in steps:
        prev_step += step
        try:
            dyn.run(step)
            atoms = dyn.atoms
            pos = atoms.get_positions()
            cell = atoms.get_cell()
            cell_max = np.max(cell)
            distance = np.linalg.norm(pos - start_pos, axis=1)
            disorder_condition = len(*np.where(distance > 2.0))
            blowup_condition = np.any(distance > 0.6*cell_max)
            
            if np.any(np.isnan(pos)) or np.any(np.abs(pos) > 1e+3) or np.unique(pos).size==1: 
                break
            elif disorder_condition > 0.5*len(pos) and not blowup_condition:
                if counter==2:
                    break
                print('Filename: ', file_name.split('/')[-1],',', 'Step :', prev_step)
                data = ase_to_ovito(atoms)
                export_file(data, f"{dump_name}_step{prev_step}.data", 'lammps/data')
                counter+=1
        except NotImplementedError as e:
            print(e)
            break

if __name__=='__main__':
    file_dir = 'crystal_files_unit/'
    file_names = listdir(file_dir)
    dump_dir = 'crystal_files_disordered/'
    dump_names = [dump_dir+f.split('.data')[0] for f in file_names]
    file_names = [file_dir+f for f in file_names]
    steps = np.ones(100, dtype=int)
    # files = file_dir + 'CNCl.data'
    # idx = file_names.index(files)
    # file_names = [file_names[idx]]
    # dump_names = [dump_names[idx]]
    for file_name, dump_name in zip(file_names, dump_names):
        print(file_name, dump_name)
        run_md_disordered(file_name, dump_name, steps)
