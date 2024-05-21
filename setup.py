# Author: Salvatore Mandra (salvatore.mandra@nasa.gov)
#
# Copyright © 2023, United States Government, as represented by the Administrator
# of the National Aeronautics and Space Administration. All rights reserved.
#
# The PySA, a powerful tool for solving optimization problems is licensed under
# the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import pybind11
import subprocess
import sys


class CMakeExtension(Extension):

    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # Check type of build
        cfg = 'Debug' if self.debug else 'Release'

        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-Dpybind11_DIR=' +
            os.environ.get('pybind11_DIR', pybind11.get_cmake_dir()),
            '-DCMAKE_BUILD_TYPE=' + cfg,
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPython_EXECUTABLE=' + sys.executable
        ]

        build_args = ['--config', cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Prepare
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp)

        # Build
        subprocess.check_call(['cmake', '--build', '.'] +
                              ['--target', ext.name] + ['-j'] + build_args,
                              cwd=self.build_temp)


# Set version
VERSION = 0.1

# Locate right path
here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get requirements from requirements.txt
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = [x.strip() for x in f.readlines()]

setup(name='pysa-dpll',
      version=VERSION,
      author='Salvatore Mandra',
      author_email='salvatore.mandra@nasa.gov',
      description='PySA-DPLL',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/nasa/PySA/tree/pysa-dpll',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Physics',
          'Programming Language :: Python :: 3 :: Only',
          'Operating System :: Unix', 'Operating System :: MacOS'
      ],
      project_urls={
          'Bug Reports': 'https://github.com/nasa/PySA/issues',
          'Source': 'https://github.com/nasa/PySA/tree/pysa-dpll',
      },
      keywords=['dpll'],
      python_requires='>=3.7',
      install_requires=install_requires,
      packages=find_packages(),
      ext_modules=[CMakeExtension('pysa_dpll_core')],
      cmdclass=dict(build_ext=CMakeBuild),
      zip_safe=False,
      scripts=['bin/pysa-dpll'])
