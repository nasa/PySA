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
project(pysa-dpll-test LANGUAGES CXX)

# Add PySA include
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add executable to compile
add_executable(test_dpll_sat.x test_dpll_sat.cpp)
add_executable(test_walk_sat.x test_walk_sat.cpp)
add_dependencies(test_dpll_sat.x pysa-branching)
add_dependencies(test_walk_sat.x pysa-branching)

# Add target to compile all tests
add_custom_target(tests
  DEPENDS test_dpll_sat.x
)

# Add target to run all tests
add_custom_target(run_tests
  COMMAND ctest
  DEPENDS tests
)

# Enable testing
enable_testing()

# Add braching test
if (MPI AND MPI_FOUND)
  add_test(NAME test_dpll_sat COMMAND ${MPIEXEC_EXECUTABLE}
          ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS}
          ${MPIEXEC_PREFLAGS} ./test_dpll_sat.x)
else ()
  add_test(NAME test_dpll_sat COMMAND ./test_dpll_sat.x)
endif ()
add_test(NAME test_walk_sat COMMAND ./test_walk_sat.x)
set_tests_properties(test_dpll_sat PROPERTIES TIMEOUT 120)
