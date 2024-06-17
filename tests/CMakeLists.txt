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

cmake_minimum_required(VERSION 3.26)

add_test(NAME pysa_stern_pretest
        COMMAND ${CMAKE_COMMAND}
        -E copy ${CMAKE_CURRENT_SOURCE_DIR}/test_n120_t8_k64_mld.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/mci_n40_t3_0_mld.txt
        .
)

if(MPI)
    set(MPI_SINGLETON ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 )
endif()

add_test(NAME test_sternx_1
        COMMAND ${MPI_SINGLETON} $<TARGET_FILE:sternx> test_n120_t8_k64_mld.txt --bench
        --sterncpp
        --max_iters 5000)
add_test(NAME test_sternx_fort
        COMMAND ${MPI_SINGLETON} $<TARGET_FILE:sternx> test_n120_t8_k64_mld.txt --bench
        --max_iters 5000
)
set_tests_properties(test_sternx_1 PROPERTIES FIXTURES_SETUP pysa_stern_pretest )
set_tests_properties(test_sternx_fort PROPERTIES FIXTURES_SETUP pysa_stern_pretest )

foreach(BLOCK_SIZE 8 16 32)
    add_test(NAME "test_sternx_b${BLOCK_SIZE}"
            COMMAND ${MPI_SINGLETON}  $<TARGET_FILE:sternx> test_n120_t8_k64_mld.txt --block-size ${BLOCK_SIZE}
            --bench
            --sterncpp
            --max_iters 5000)
    set_tests_properties("test_sternx_b${BLOCK_SIZE}" PROPERTIES FIXTURES_SETUP pysa_stern_pretest )
endforeach()


foreach(PARAMS IN ITEMS "1;8" "1;16" "1;32" "2;8" "4;8" "8;8" "32;32")
    list(GET PARAMS 0 M)
    list(GET PARAMS 1 L)
    add_test(NAME "test_sternx_m${M}l${L}"
            COMMAND ${MPI_SINGLETON} $<TARGET_FILE:sternx> test_n120_t8_k64_mld.txt -m ${M} -l ${L}
            --bench
            --sterncpp
            --max_iters 5000)
    set_tests_properties("test_sternx_m${M}l${L}" PROPERTIES FIXTURES_SETUP pysa_stern_pretest )
endforeach()

set_tests_properties(test_sternx_m32l32 PROPERTIES WILL_FAIL TRUE)

if(MPI)
    add_test(NAME test_sternx_mpi
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
            ${MPIEXEC_MAX_NUMPROCS}
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:sternx> test_n120_t8_k64_mld.txt
            --bench
            --sterncpp
            --max_iters 50000
            )
    add_test(
        NAME test_sternx_mpi_fort
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
        ${MPIEXEC_MAX_NUMPROCS}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:sternx> test_n120_t8_k64_mld.txt
        --bench
        --max_iters 50000
    )
    set_tests_properties(test_sternx_mpi PROPERTIES FIXTURES_SETUP pysa_stern_pretest )
    set_tests_properties(test_sternx_mpi_fort PROPERTIES FIXTURES_SETUP pysa_stern_pretest )
    # prevent error on macos
    set_tests_properties(test_sternx_mpi PROPERTIES ENVIRONMENT TMPDIR=/tmp)
    set_tests_properties(test_sternx_mpi_fort PROPERTIES ENVIRONMENT TMPDIR=/tmp)
endif()

add_test(NAME test_mldpt
        COMMAND mldpt mci_n40_t3_0_mld.txt 10000 --beta-min 0.1 --beta-max 1.0 --num-temps 16
)
set_tests_properties(test_mldpt PROPERTIES FIXTURES_SETUP pysa_stern_pretest )