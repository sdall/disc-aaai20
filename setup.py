import os
import sys
import pybind11
from setuptools import setup, Extension

cpp_args = ['-std=c++17', '-O3', '-march=native']
cpp_libs = ['pthread', 'gomp']

ext_modules = [
    Extension(
        'disc',
        ['./src/pybind11/PyDisc.cpp'],
        include_dirs=['./include', pybind11.get_include(True)],
        language='c++',
        libraries=cpp_libs,
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='disc',
    version='0.2.0',
    author='Sebastian Dalleiger',
    author_email='sdalleig@mpi-inf.mpg.de',
    license="MIT",
    description='Discover the pattern composition, that is (i) a partitioning of a dataset into components that have significantly differently distributed patterns; and (ii) describe the partitioning using characteristic patterns or patterns shared among sets of components.',
    ext_modules=ext_modules,
    install_requires=['pybind11', 'numpy'],
)
