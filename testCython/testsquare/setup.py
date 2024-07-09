#standard stuff:
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

#Mystuff:
import os
os.environ["CXX"]="icpc"


sourcefiles=['csquareit.pyx','squareit.cpp']

extensions=Extension('csquareit',
	sourcefiles,
	language='c++',
	extra_compile_args=['-fopenmp'],
	extra_link_args=['-fopenmp'])

setup(
	cmdclass={'build_ext':build_ext},
	ext_modules=cythonize(extensions)
	)
