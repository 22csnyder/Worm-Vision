from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

extensions=Extension('h-world',['helloworld.pyx'],language='c++')
setup(
	cmdclass={'build_ext': build_ext},
	ext_modules=cythonize(extensions)
	)
