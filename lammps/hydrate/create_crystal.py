from ase import Atoms
import ase
from ovito.io import export_file
from ovito.io.ase import ase_to_ovito
d = 1.1
co = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)])

# http://rruff.geo.arizona.edu/AMS/minerals/Methane-hydrate
from ase.spacegroup import crystal
# Create ice lh
# Pm3n
a = 11.877
hydrate_lattice_1 = crystal(('O', 'O', 'O'),
                       basis=[(0, .30860, .11683), (.18345, .18345, .18345), (0, .5, .25)],
                       spacegroup=223,
                       cellpar=[a, a, a, 90, 90, 90])

methane_clathrated_1 = crystal(('C', 'C'),
                       basis=[(0, 0, 0), (.25, .5, 0)],
                       spacegroup=223,
                       cellpar=[a, a, a, 90, 90, 90])

full_clathrate_1_lattice = hydrate_lattice_1 + methane_clathrated_1


# Bernal J, Fowler R Journal of Chemical Physics 1 (1933) 515-548 A Theory of Water and Ionic Solution, with Particular Reference to Hydrogen and Hydroxyl Ions
ice_lh = crystal(('O', 'O'),
                basis=[(1/3, 0, .0625), (2/3,    0, .9375)],
                spacegroup = 185,
                cellpar=[7.82, 7.82, 7.36, 90, 90, 120],
                size=5)
    

 
# ice_lh.rotate(-90, [1,0,0], rotate_cell=False)

ice_lh = ase_to_ovito(ice_lh)
export_file(ice_lh, 'ice_ih.data', 'lammps/data')
