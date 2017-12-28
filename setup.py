from setuptools import setup, Extension, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import shutil

import numpy

import os

# Check compiler - default is g++
CXX = os.environ.get('CXX', 'g++')

# INTEL ARGUMENTS
USE_INTEL = CXX=='icpc'
MKLROOT = os.environ.get('MKLROOT', False)
intel_compile_args = [
    "-std=c++11", '-pthread', '-O3','-fopenmp', '-simd', '-qopenmp', '-xHost',
    '-DUSEMKL', '-DMKL_ILP64', '-I{}/include'.format(MKLROOT)
]
intel_link_args = [
    '-std=c++11', '-pthread', '-fopenmp', '-qopenmp',
    '-Wl,--start-group', '{}/lib/intel64/libmkl_intel_ilp64.a'.format(MKLROOT),
    '{}/lib/intel64/libmkl_sequential.a'.format(MKLROOT),
    '{}/lib/intel64/libmkl_core.a'.format(MKLROOT), '-Wl,--end-group',
    '-lpthread', '-lm', '-ldl'
]
intel_libs = ['moma', 'm', 'dl', 'pthread']


# GNU ARGUMENTS
gnu_compile_args = ["-std=c++11", '-pthread', '-O3', '-fopenmp']
gnu_link_args = [
    '-std=c++11', '-pthread', '-fopenmp'
]
gnu_libs = ['moma']

setup(
    name='magpy',
    version='0.1.dev2',
    packages=find_packages(),
    ext_modules= cythonize(Extension(
        name='magpy.core',
        sources=["magpy/core.pyx"],
        language='c++',
        extra_compile_args= intel_compile_args if USE_INTEL else gnu_compile_args,
        extra_link_args= intel_link_args if USE_INTEL else gnu_link_args,
        libraries=intel_libs if USE_INTEL else gnu_libs,
        library_dirs=['.', '{}/lib/intel64'.format(MKLROOT)],
        include_dirs=[numpy.get_include(), './include']
    ))
)
