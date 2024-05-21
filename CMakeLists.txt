# Author: Salvatore Mandra (salvatore.mandra@nasa.gov)
#         Humberto Munoz-Bauza (humberto.munozbauza@nasa.gov)
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
project(pysa-dpll LANGUAGES CXX)

# Set the C++ standard to c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add options
add_compile_options(-O3 -Wall -Wpedantic -Wfatal-errors)

# Use MPI
option(MPI "Build with MPI" OFF)

# Find MPI package
find_package(MPI)

# If MPI exists, compile the code using MPI
if (MPI AND MPI_FOUND)
  message(STATUS "MPI Enabled")
  add_compile_definitions(USE_MPI)
  include_directories(${MPI_CXX_HEADER_DIR})
  link_libraries(MPI::MPI_CXX)
endif ()

# Find pthreads
find_package(Threads REQUIRED)

# Link threads
link_libraries(Threads::Threads)

# Add external files
include(ExternalProject)

ExternalProject_Add(
        pysa-branching
        GIT_REPOSITORY https://github.com/nasa/pysa.git
        GIT_TAG pysa-branching
        GIT_SHALLOW 1
        GIT_PROGRESS 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        TEST_COMMAND ""
)
ExternalProject_Get_Property(pysa-branching install_dir)
include_directories(${install_dir}/src/pysa-branching/include)

# Add executable, tests, and examples
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(include/pysa/python)
