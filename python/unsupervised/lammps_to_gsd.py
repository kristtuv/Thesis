from ovito.io import export_file, import_file
import sys
input = str(sys.argv[1])
output = input.rsplit('.', maxsplit=1)[0] + '.gsd'
pipeline = import_file(input)
print(dir(pipeline))
export_file(pipeline, output, "gsd", columns=['Position.X'])

