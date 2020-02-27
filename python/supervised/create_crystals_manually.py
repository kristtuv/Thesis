import ase.spacegroup
import ase
import numpy as np
from ase.io import write 
from ovito.io import export_file
from ovito.io.ase import ase_to_ovito

#rruff.geo.arizona.edu/AMS

my_crystals = {

    "methane_hydrate_I" : 
    {
        "symbols" : ("O", "O", "O", "C", "C"),
        "spacegroup" : 223,
        "cellpar" : [12.03, 12.03, 12.03, 90, 90, 90],
        "basis" : [(.1841,.1841,.1841),(0,.3100,.1154),(0,.5,.25),(0,0,0),(.25,.5,0)],
        # "size": (5, 5, 5)
        # "size": (4, 4, 4)
        # "size": (2, 2, 2)
    },

    "methane_hydrate_II" : 
    {
        "symbols" : ["O", "O", "O", "C"], 
        "spacegroup" : 227,
        "cellpar" : [17.092, 17.092, 17.092, 90, 90, 90],
       "basis" : [  (.1822, .1822, .3719), 
                    (.2196, .2196, .2196), 
                    (.125,  .125,  .125),
                    (.375,  .375, .375)
                    ],
        "setting" : 2,
        # "size": (4, 4, 4)
    },

    "methane_hydrate_H" : 
    {
        "symbols" : ["O", "O", "O", "O",  "C", "C", "C"],
        "spacegroup" : 191,
        "cellpar" : [11.9100 , 11.9100, 9.8940, 90, 90, 120],
        "basis" : [ (0., 0.38, 0.125), 
                    (0.22,0.44,0.25),
                    (0.125,0.25,0.5),
                    (0.333,-0.333,-0.375 ),
                    (0.5,0.5,0.5),
                    (0.3333, -0.3333, 0.0),
                    (0,0,0)],
        # "size": (4, 4, 4),
        'onduplicates': 'replace',
        'symprec': 0.002,
    },
}

def create_crystal(crystal_info, **kwargs):
    return ase.spacegroup.crystal(**crystal_info, **kwargs)

if __name__=='__main__':
    dump_dir = 'crystal_files_unit_poscar/'
    steps = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    for key, value in my_crystals.items():
        dump_name = dump_dir+key
        print('Dump_name: ', dump_name)
        crystal = create_crystal(value)
        n_particles = crystal.get_number_of_atoms()
        min_particles_after_replicate = 500
        replicate = int(np.ceil((min_particles_after_replicate / n_particles)**(1/3)))
        crystal = create_crystal(value, size=replicate)
        ase.geometry.get_duplicate_atoms(crystal, delete=True)
        n_particles = crystal.get_number_of_atoms()
        print('Crystal: ', key, '      ',  'Number of particles: ', n_particles)
        data = ase_to_ovito(crystal)
        export_file(data, f'{dump_name}.data', 'lammps/data')
