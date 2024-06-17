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

cmake_minimum_required(VERSION 3.20)
project(stern)

option(MPI "Build with MPI" OFF)
option(AVX2 "Enable AVX2" ON)
option(USE_SIMDE "Enable using the simde library" ON)
option(PYMODULES "Build Python Modules for PySA" ON)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
enable_language(Fortran)

find_package(Git REQUIRED)
# Alternatively, run git submodule update --init --recursive after checkout
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/lib/libmld/lib/simde/COPYING)
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
          COMMAND_ERROR_IS_FATAL ANY)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(MPI)
  message("MPI Enabled")
  find_package(MPI REQUIRED)
else()
  message("MPI Disabled")
endif()


add_subdirectory(lib/libmld)
add_subdirectory(src)

## Add tests
add_subdirectory(./tests)