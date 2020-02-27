from distutils.core import setup
import sys
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize(
        "adjacencybuilder.pyx",
        compiler_directives={'language_level' : sys.version_info[0]},
        ),
    include_dirs = [np.get_include()]
)
