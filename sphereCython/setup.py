from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import os
os.environ["CXX"]="icpc"

sourcefiles=['sphhar.pyx','harm.cpp']

extensions=Extension('sphhar',
	sourcefiles,
	language='c++',
	extra_compile_args=['-fopenmp','-I/opt/apps/intel13/gsl/1.16/x86_64/include'],
	extra_link_args=['-L/opt/apps/intel13/gsl/1.16/x86_64/lib',
			'-lgsl','-lgslcblas','-lm'])
	
setup(
	cmdclass={'build_ext':build_ext},
	ext_modules=cythonize(extensions)
	)
	
	
