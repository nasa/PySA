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

cmake_minimum_required(VERSION 3.21)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(stern_fort)

find_package(Boost 1.46.0
        REQUIRED COMPONENTS program_options chrono timer)

add_library(sterncpp STATIC wf.cpp stern.cpp)
add_executable(sternx main.cpp)
add_executable(mldpt pt.cpp )

## Set Compile Options
if (CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
    target_compile_options(sternx PRIVATE
            "$<$<AND:$<CONFIG:Release>,
     $<COMPILE_LANGUAGE:CXX>>:-ip;-ipo;-qopt-report=5;
                              -qopt-report-phase=all;-O3;-xAVX>")
    target_link_options(sternx PRIVATE
            "$<$<CONFIG:Release>:-ip;-ipo;-qopt-report=5;
                         -qopt-report-phase=all;-O3;-xAVX>")
    if(AVX2)
        target_compile_options(sternx PRIVATE -xCORE-AVX2)
        target_link_options(sternx PRIVATE -xCORE-AVX2)
    endif()
endif ()
target_compile_options(sternx PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wpedantic>")
target_compile_options(mldpt PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wpedantic>")

## Set include directories and compile definitions
if(USE_SIMDE)
    include_directories(../lib/libmld/lib/simde/)
    add_compile_definitions(USE_SIMDE)
endif ()
if(MPI)
    target_compile_definitions(sterncpp PUBLIC USEMPI)
    target_compile_definitions(sternx PUBLIC USEMPI)
endif()

target_include_directories(sterncpp PUBLIC ../lib/libmld/include
        ../lib/simde
        ../include/)
target_include_directories(sternx PUBLIC ../lib/libmld/include
../include/)
target_include_directories(mldpt PUBLIC ../lib/libmld/include
        ../lib/simde
        ../include/)

## Linkage
if(MPI)
    target_link_libraries(sterncpp MPI::MPI_CXX)
    target_link_libraries(sternx MPI::MPI_CXX)
endif()
target_link_libraries(sterncpp Boost::chrono Boost::timer)
target_link_libraries(sternx stern sterncpp Boost::program_options)
target_link_libraries(mldpt Boost::program_options Boost::timer)

if(MPI AND CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
    target_link_libraries(sternx mpi_f08_intel)
endif()

## Add Python module
if(PYMODULES)
    add_subdirectory(bindings)
endif ()