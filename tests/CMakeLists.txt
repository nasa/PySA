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
project(pysa-branching-tests LANGUAGES CXX)

# Set the C++ standard to c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add options
add_compile_options(-Ofast -Wall -Wpedantic -Wfatal-errors)

# Find MPI package
find_package(MPI)

# Use MPI
option(MPI "Build with MPI" OFF)

# Add executable to compile
add_executable(test_archive.x test_archive.cpp test_archive.hpp)
add_executable(test_branching.x test_branching.cpp test_branching.hpp)

# Add include folders
target_include_directories(test_archive.x PUBLIC ../include)
target_include_directories(test_branching.x PUBLIC ../include)

# Add target to compile all tests
add_custom_target(tests
  DEPENDS test_archive.x
  DEPENDS test_branching.x
)

# Add target to run all tests
add_custom_target(run_tests
  COMMAND ctest
  DEPENDS tests
)

# If MPI exists, compile the code using MPI
if (MPI AND MPI_FOUND)
  message(STATUS "MPI Enabled")
  target_compile_definitions(test_branching.x PUBLIC USE_MPI)
  target_include_directories(test_branching.x PUBLIC ${MPI_CXX_HEADER_DIR})
  target_link_libraries(test_branching.x MPI::MPI_CXX)
endif ()

# Enable testing
enable_testing()

# Add archive test
add_test(NAME test_archive COMMAND ./test_archive.x)

# Add braching test
if (MPI AND MPI_FOUND)
  add_test(NAME test_branching COMMAND mpirun -n 3 ./test_branching.x)
else ()
  add_test(NAME test_branching COMMAND ./test_branching.x)
endif ()
set_tests_properties(test_branching PROPERTIES TIMEOUT 200)
