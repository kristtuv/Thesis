from create_crystals_from_poscar import replicate_cell
# pipe = import_file()
# data = pipe.compute()
# export_file(data, 'cubic.data', 'lammps/data') 

file_name = '1541503.cif'
dump_name = file_name.replace('.cif', '')

replicate_cell(file_name, dump_name)
