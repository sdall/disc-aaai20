import os
import sys
import pybind11
import platform
from setuptools import setup, Extension

cpp_args = []
cpp_libs = []

if platform.system() == 'Windows':
    cpp_args = ['/std:c++latest', '/MD', '/O2', '/Ob2', '/DNDEBUG', '/MT', '/openmp']
else:
    cpp_args = ['-std=gnu++17', '-O3', '-march=native', '-fopenmp']
    cpp_libs = ['gomp', 'tbb']

ext_modules = [
    Extension(
        'disc',
        ['./src/pybind11/PyDisc.cpp'],
        include_dirs=['./include','./src', pybind11.get_include(True)],
        language='c++',
        libraries=cpp_libs,
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='disc',
    version='0.4.0',
    author='Sebastian Dalleiger',
    author_email='sdalleig@mpi-inf.mpg.de',
    license="MIT",
    description='Discover the pattern composition, that is (i) a partitioning of a dataset into components that have significantly differently distributed patterns; and (ii) describe the partitioning using characteristic patterns or patterns shared among sets of components.',
    ext_modules=ext_modules,
    install_requires=['pybind11', 'numpy'],
)
