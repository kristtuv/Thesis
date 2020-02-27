from ovito.io import import_file, export_file
from ovito.modifiers import CombineDatasetsModifier
from ovito.data import DataCollection
from os import listdir

filedir = '400quasi'
files = listdir(filedir)
modifier = CombineDatasetsModifier()
print(dir(modifier))
exit()
data = DataCollection()
# for f in files:
#     data.apply( = import_file(filedir+f).compute(-1)


print(files)
