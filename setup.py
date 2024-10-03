# Author: Humberto Munoz Bauza (humberto.munozbauza@nasa.gov)
#
# Copyright © 2024, United States Government, as represented by the Administrator
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

import sys
import setuptools
from setuptools import setup, Extension, find_packages
from pathlib import Path
from cmake_build_extension import CMakeExtension, BuildExtension

init_py = ""

setup(
    ext_modules=[
        CMakeExtension(
            name="pysa_walksat",
            install_prefix="pysa_walksat",
            cmake_depends_on=["pybind11"],
            source_dir=str(Path(__file__).parent.absolute()),
            cmake_configure_options=[
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython_ROOT_DIR={Path(sys.prefix)}",
                f"-DPYMODULES=ON"
            ],
            cmake_component="bindings",
            write_top_level_init=init_py  
        ),
    ],
    cmdclass=dict(build_ext=BuildExtension),
    packages=find_packages("./src"),
    package_dir={
            '': 'src'
    },
)