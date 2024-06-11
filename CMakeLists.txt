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
add_subdirectory(lib/libmld)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(MPI)
  message("MPI Enabled")
  find_package(MPI REQUIRED)
else()
  message("MPI Disabled")
endif()

find_package(Boost 1.46.0 
             REQUIRED COMPONENTS program_options chrono timer)

add_library(stern src/stern.f90
                  src/sternalgs.f90
                  src/sternc.f90
                  src/lincode.f90
                  src/randutil.f90
                  src/mathutil.f90)
target_include_directories(stern PRIVATE ${MPI_Fortran_HEADER_DIR})

add_executable(sternx src/main.cpp 
                      src/wf.cpp 
                      include/sternx/stern.h 
                      ./src/stern.cpp)
add_executable(mldpt src/pt.cpp )

if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(stern PRIVATE ${MPI_Fortran_COMPILE_OPTIONS} -cpp)

elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  target_compile_definitions(stern PUBLIC IFORT)
  target_compile_options(stern PRIVATE ${MPI_Fortran_COMPILE_OPTIONS} -fpp)
  target_include_directories(stern PRIVATE ${MPI_Fortran_MODULE_DIR}/intel)
  target_compile_options(stern PRIVATE  
    "$<$<AND:$<CONFIG:Release>,
     $<COMPILE_LANGUAGE:Fortran>>:-ip;-ipo;-qopt-report=5;
                                  -qopt-report-phase=all;-O3;-xAVX>")
  target_compile_options(sternx PRIVATE  
    "$<$<AND:$<CONFIG:Release>,
     $<COMPILE_LANGUAGE:CXX>>:-ip;-ipo;-qopt-report=5;
                              -qopt-report-phase=all;-O3;-xAVX>")
  target_link_options(sternx PRIVATE 
    "$<$<CONFIG:Release>:-ip;-ipo;-qopt-report=5;
                         -qopt-report-phase=all;-O3;-xAVX>")

  if(AVX2)
    target_compile_options(stern PRIVATE 
      "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:Fortran>>:-xCORE-AVX2>" )
    target_compile_options(sternx PRIVATE -xCORE-AVX2)
    target_link_options(sternx PRIVATE -xCORE-AVX2)
  endif()

endif()

if(USE_SIMDE)
  include_directories(./lib/libmld/lib/simde/)
  add_compile_definitions(USE_SIMDE)
endif ()

if(MPI)
  target_compile_definitions(stern PUBLIC USEMPI)
  target_link_libraries(stern MPI::MPI_Fortran)
endif()

target_compile_options(stern PRIVATE
        "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:Fortran>>:-fcheck=all>")
target_compile_options(stern PRIVATE "$<$<COMPILE_LANGUAGE:Fortran>:-Wall;-Wextra;>")
target_compile_options(sternx PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wpedantic>")
target_compile_options(mldpt PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wpedantic>")

target_include_directories(sternx PUBLIC ./lib/libmld/include 
                                         ./lib/simde 
                                         ./include/)
target_include_directories(mldpt PUBLIC ./lib/libmld/include
                                        ./lib/simde
                                        ./include/)

target_link_libraries(sternx stern
                             libmld
                             Boost::program_options
                             Boost::chrono Boost::timer)
target_link_libraries(mldpt libmld
                            Boost::program_options Boost::timer)
if(MPI)
  target_link_libraries(sternx MPI::MPI_CXX)
endif()
if(MPI AND CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  target_link_libraries(sternx mpi_f08_intel)
endif()

## Add tests
add_subdirectory(./tests)