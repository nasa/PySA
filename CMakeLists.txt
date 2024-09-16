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

cmake_minimum_required(VERSION 3.21)
project(pysa_cdcl)
enable_testing()

include_directories(include)
set(CMAKE_CXX_STANDARD 17)
if(SATTRACE)
    add_compile_definitions("SATTRACE")
endif ()
add_library(cdcl_lib STATIC cdcl.cpp)
add_executable(test_cdcl ./tests/cdcl_tests.cpp cdcl.cpp)
add_executable(pysa_cdcl main.cpp cdcl.cpp)

add_test(NAME test_cdcl COMMAND test_cdcl)
## Python bindings

if(PYBINDINGS)
    message(" ** Python bindings for PySA-CDCL are a beta feature. **")
    # Use FindPython or FindPython3 module
    find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
    # Check for pybind11
    find_package(pybind11 REQUIRED)
    # Build the Python extension bindings module
    pybind11_add_module(pysa_cdcl_ext bindings.cpp)
    target_link_libraries(pysa_cdcl_ext PUBLIC cdcl_lib)
    # make sure object name matches the defined module name
    set_target_properties(pysa_cdcl_ext PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pysa2_cdcl
            OUTPUT_NAME "bindings")
    # Installs only the Python extension object when the "bindings" component is specified in setup.py
    install(
            TARGETS pysa_cdcl_ext
            DESTINATION ${CMAKE_INSTALL_PREFIX}
            COMPONENT bindings
    )
endif ()