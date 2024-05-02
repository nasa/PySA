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
project(pysa-dpll-examples LANGUAGES CXX)

# Add executable to compile
add_executable(simple.x simple.cpp)
add_dependencies(simple.x pysa-branching)

# Add target to compile all examples
add_custom_target(examples
  DEPENDS simple.x
)

# Add target to run all examples
add_custom_target(run_examples
  COMMAND ./simple.x
  DEPENDS examples
)

# If MPI exists, compile the code using MPI
if (MPI AND MPI_FOUND)
  message(STATUS "MPI Enabled")
  target_compile_definitions(simple.x PUBLIC USE_MPI)
  target_include_directories(simple.x PUBLIC ${MPI_CXX_HEADER_DIR})
  target_link_libraries(simple.x MPI::MPI_CXX)
endif ()
