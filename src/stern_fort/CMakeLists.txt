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


add_library(stern STATIC stern.f90
        sternalgs.f90
        sternc.f90
        lincode.f90
        randutil.f90
        mathutil.f90)
target_include_directories(stern PRIVATE ${MPI_Fortran_HEADER_DIR})

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
    if(AVX2)
        target_compile_options(stern PRIVATE
                "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:Fortran>>:-xCORE-AVX2>" )
    endif()
endif()

if(MPI)
    target_compile_definitions(stern PUBLIC USEMPI)
    target_link_libraries(stern MPI::MPI_Fortran)
endif()

target_compile_options(stern PRIVATE
        "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:Fortran>>:-fcheck=all>")
target_compile_options(stern PRIVATE "$<$<COMPILE_LANGUAGE:Fortran>:-Wall;-Wextra;>")