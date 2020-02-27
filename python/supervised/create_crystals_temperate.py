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
from ase.calculators.colloidal import Colloidal
from ase.calculators.harmonic import Harmonic
import numpy as np
from ovito.modifiers import CoordinationAnalysisModifier
from util import cutoff_finder

coordination = CoordinationAnalysisModifier(cutoff=10, number_of_bins=100)

def run_md(file_name=None, dump_name=None, steps=None, crystal=None):
    if dump_name.startswith('disordered'):
        print('Warning: Your are dumping in disordered directory')
    if crystal==None:
        crystal = import_file(file_name).compute().to_ase_atoms()
 
    data = ase_to_ovito(crystal)
    data.apply(coordination)
    cut = cutoff_finder(data)
    positions = crystal.get_positions()
    calc = Harmonic(positions)
    crystal.set_calculator(calc)
    start_pos = crystal.get_positions()
    # dyn = Langevin(crystal, 2*units.fs, 300*units.fs, 0.002)
    dyn = Langevin(crystal, 0.1, 0.4, 0.02)
    prev_step=0
    for step in steps:
        prev_step += step
        print('Filename: ', file_name.split('/')[-1],',', 'Step :', prev_step)
        try:
            dyn.run(step)
            atoms = dyn.atoms
            pos = atoms.get_positions()
            data = ase_to_ovito(atoms)
            export_file(data, f"{dump_name}_step{prev_step}.data", 'lammps/data')
            # if np.all(np.linalg.norm(pos - start_pos, axis=1) < 0.25*cut ):
            #     data = ase_to_ovito(atoms)
            #     export_file(data, f"{dump_name}_step{prev_step}.data", 'lammps/data')
            # else:
            #     break
            if np.all(np.linalg.norm(pos - start_pos, axis=1) < 0.05*cut ):
                data = ase_to_ovito(atoms)
                export_file(data, f"{dump_name}_step{prev_step}.data", 'lammps/data')
            else:
                break
        except NotImplementedError as e:
            print(e)
            break

if __name__=='__main__':
    file_dir = 'crystal_files_unit_poscar/'
    file_names = listdir(file_dir)
    # file_names = [f for f in file_names if 'a3b' in f]
    # dump_dir = 'test/'
    dump_dir = 'crystal_files_harmonic_newbreak/'
    dump_names = [dump_dir+f.split('.data')[0] for f in file_names]
    file_names = [file_dir+f for f in file_names]
    steps = np.ones(300, dtype=int)
    for file_name, dump_name in zip(file_names, dump_names):
        run_md(file_name, dump_name, steps)
        # if file_name.startswith(file_dir+'methane'):
        #     print(file_name, dump_name)
        #     run_md(file_name, dump_name, steps)
        #     exit()

