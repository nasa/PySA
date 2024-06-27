# Author: Humberto Munoz Bauza (humberto.munozbauza@nasa.gov)
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

from setuptools import setup, find_packages
import os
from pathlib import Path
import sys

from cmake_build_extension import CMakeExtension, BuildExtension

# Set version
version = '0.0.1'

# Locate right path
here = os.path.abspath(os.path.dirname(__file__))


def readme():
    with open(os.path.join(here, 'README.md')) as f:
        return f.read()


# Get requirements from requirements.txt
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = [x.strip() for x in f.readlines()]

setup(name='McEliece',
      version=version,
      packages=find_packages(include=["mceliece"]),
      python_requires='>=3.8',
      install_requires=install_requires,
      extras_require={'dev': ['pytest']},
      ext_modules=[
        CMakeExtension(
            name="mceliece-tseytin",
            install_prefix="mceliece/tseytin",
            cmake_depends_on=["pybind11"],
            source_dir=str(Path(__file__).parent.absolute()/"mceliece"/"tseytin"),
            cmake_configure_options=[
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython_ROOT_DIR={Path(sys.prefix)}"
            ],
            cmake_component="bindings",
        ),
    ],
    cmdclass=dict(build_ext=BuildExtension)
)
