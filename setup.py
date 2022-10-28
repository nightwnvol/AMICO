from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from numpy import get_include

# import details from amico/info.py
import sys
sys.path.insert(0, './amico/')
import info

sources = ['amico/models.pyx']
include_dirs = [
      'spams',
      'spams/decomp',
      'spams/dictLearn',
      'spams/linalg',
      'spams/prox',
      'nnls',
      get_include()
]
libraries = []
library_dirs = []
extra_compile_args = []
extra_link_args = []

if sys.platform.startswith('win32'):
      include_dirs.extend(['C:/Users/clori/Desktop/install_spams/OpenBLAS-0.3.20-x64/include']) # NOTE only for tests
      libraries.extend(['libopenblas'])
      library_dirs.extend(['C:/Users/clori/Desktop/install_spams/OpenBLAS-0.3.20-x64/lib']) # NOTE only for tests
      extra_compile_args.extend(['-std:c11'])
      extra_link_args.extend([])
if sys.platform.startswith('linux'):
      include_dirs.extend([])
      libraries.extend(['stdc++', 'blas', 'lapack'])
      library_dirs.extend([])
      extra_compile_args.extend(['-std=c++11'])
      extra_link_args.extend([])
if sys.platform.startswith('darwin'):
      include_dirs.extend([])
      libraries.extend(['stdc++', 'blas', 'lapack'])
      library_dirs.extend([])
      extra_compile_args.extend(['-std=c++11'])
      extra_link_args.extend([])

extensions = [
      Extension(
            'amico.models',
            sources=sources,
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
      )
]

setup(name=info.NAME,
      version=info.VERSION,
      description=info.DESCRIPTION,
      long_description=info.LONG_DESCRIPTION,
      author=info.AUTHOR,
      author_email=info.AUTHOR_EMAIL,
      url=info.URL,
      license=info.LICENSE,
      packages=find_packages(),
      install_requires=['packaging', 'wheel', 'numpy>=1.12', 'scipy>=1.0', 'dipy>=1.0', 'spams>=2.6.5.2', 'tqdm>=4.56.0', 'joblib>=1.0.1', 'Cython>=0.29'],
      package_data={'': ['*.bin', 'directions/*.bin', '*.dll']}, # NOTE only for tests
      ext_modules=cythonize(extensions))
