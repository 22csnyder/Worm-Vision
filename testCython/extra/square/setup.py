from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


sourcefiles=['csquareit.pyx','squareit.cpp']

extensions=Extension('csquareit',sourcefiles,language='c++')

setup(
	cmdclass={'build_ext':build_ext},
	ext_modules=cythonize(extensions)
	)
