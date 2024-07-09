from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import os

os.environ["CXX"]="icpc"
sourcefiles=['cy_sphere_iter.pyx','iteration.cpp','getinfo.cpp']

extensions=Extension('cy_sphere_iter',
	sourcefiles,
	language='c++',
	extra_compile_args=['-fopenmp','-std=c++0x'],
	extra_link_args=[])

setup(
	cmdclass={'build_ext':build_ext},
	ext_modules=cythonize(extensions)
	)

