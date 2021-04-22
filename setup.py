import os
import sys
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

DISTNAME = 'disc'
DESCRIPTION = 'Discover and Describe Diverging Data Partitions'
AUTHOR = 'Sebastian Dalleiger'
AUTHOR_EMAIL = 'sdalleig@mpi-inf.mpg.de'
LICENSE = 'MIT'
VERSION = "0.2.2"
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

        # subprocess.check_call(['meson', ext.src, "--prefix", t], cwd=self.build_temp)
        # subprocess.check_call(['meson', 'compile'], cwd=self.build_temp)
        # subprocess.check_call(['meson', 'install'], cwd=self.build_temp)

        subprocess.check_call(['cmake',
                               '-H{}'.format(ext.src),
                               '-B{}'.format(self.build_temp),
                               '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
                               "-DCMAKE_INSTALL_PREFIX={}".format(t),
                               "-DCMAKE_BUILD_TYPE=Release"])
        subprocess.check_call(['cmake',
                               '--build',
                               self.build_temp,
                               '--target', 'install'])
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
    install_requires=['pybind11', 'numpy'],
)
