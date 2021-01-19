import os, sys, shutil, subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import cmake

DISTNAME = 'disc'
DESCRIPTION = 'Discover and Describe Diverging Data Partitions'
AUTHOR = 'Sebastian Dalleiger'
AUTHOR_EMAIL = 'sdalleig@mpi-inf.mpg.de'
LICENSE = 'MIT'
VERSION = "0.2.1"
URL = 'https://github.com/sdalleig/disc-aaai20'
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'Intended Audience :: Data Scientists',
               'License :: MIT',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Programming Language :: C++',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS']

with open('README.md') as readme_file:
    LONG_DESCRIPTION = readme_file.read()

class Ext(Extension):
    def __init__(self, name, src=''):
        Extension.__init__(self, name, sources=[])
        self.src = os.path.abspath(src)


class Build(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # required for auto-detection of auxiliary "native" libs
        t = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not t.endswith(os.path.sep):
            t += os.path.sep

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        cmake_args = ['-DPYTHON_EXECUTABLE=' + sys.executable,
                      "-DCMAKE_BUILD_TYPE=Release",
                      "-DCMAKE_INSTALL_PREFIX={}".format(t)]

        cmake_bin = cmake.CMAKE_BIN_DIR + os.path.sep + 'cmake'

        subprocess.check_call([cmake_bin, '-H{}'.format(ext.src), '-B{}'.format(self.build_temp)] + cmake_args)
        subprocess.check_call([cmake_bin, '--build', self.build_temp,'--target', 'install'])

        shutil.rmtree(self.build_temp)

setup(
    name=DISTNAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license=LICENSE,
    version=VERSION,
    url=URL,
    download_url=URL,
    classifiers=CLASSIFIERS,
    options={"bdist_wheel": {"universal": True}},
    ext_modules=[Ext('disc')],
    cmdclass=dict(build_ext=Build),
    zip_safe=False,
    install_requires=['cmake', 'pybind11', 'numpy'],
)