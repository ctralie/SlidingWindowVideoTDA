from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Clique Algorithms',
  ext_modules = cythonize("CliqueAlgorithms.pyx"),
)

