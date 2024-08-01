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
project(pysa-walksat-main LANGUAGES CXX)

# add includes
include_directories(../include)

add_library(walk-sat walk-sat.cpp)

# Add executable to compile
add_executable(walk-sat.x main.cpp)
target_link_libraries(walk-sat.x walk-sat )
