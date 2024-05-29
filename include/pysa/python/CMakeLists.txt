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

# Set the minimum required version for cmake
cmake_minimum_required(VERSION 3.5)

# Set the project name
project(pysa-dpll-python LANGUAGES CXX)

# Set the C++ standard to c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use FindPython or FindPython3 module
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)

# Check pybind11
find_package(pybind11 REQUIRED)

# Get all source files
file(GLOB_RECURSE SOURCES "./*.cpp")

# Initialize PyBind11 library
pybind11_add_module(pysa_dpll_core ${SOURCES})
target_include_directories(pysa_dpll_core PRIVATE ../../)
add_dependencies(pysa_dpll_core pysa-branching)
