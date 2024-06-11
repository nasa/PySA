/*
Author: Humberto Munoz Bauza (humberto.munozbauza@nasa.gov)

Copyright © 2023, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The PySA, a powerful tool for solving optimization problems is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
*/

#ifndef STERN_STERN_H
#define STERN_STERN_H
#include <cstdint>
#include "libmld/mld.h"

struct sternc_opts
{
    /// (F) Problem defines a parity check matrix if true. Otherwise it defines the generator matrix transpose.
    int8_t parcheck=0;
    /// (F, C) Benchmark mode: If true, perform all max_iters iterations and
    /// calculate the success probability and time-to-solution.
    /// If false, stop as soon as the solution is found.
    int8_t bench=0;
    /// (C, Experimental) Allow a collision with a Hamming weight of 1 out of l rows
    /// to trigger a sum across all n-k rows in the standard Stern algorithm.
    /// Ignored in the heavy Stern algorithm.
    int8_t test_hw1=false;
    /// (F) Number of columns in the matrix. If parcheck>0, this is the number of colmuns of the
    /// parity check matrix n. Otherwise, it is the number of columns of the generator matrix transpose k.
    int32_t nvars=0;
    /// (F) Number of rows in the matrix. If parcheck>0, this is the number of rows of the
    /// parity check matrix n-k. Otherwise, it is the number of rows of the generator matrix transpose n.
    int32_t nclauses=0;
    /// (F, C) Error weight
    int32_t t=-1;
    /// (F, C) Maximum number of iterations of the Stern algorithm
    int32_t max_iters;
    /// (F, C) Size of collision set to check. (Ignored in the heavy Stern algorithm.)
    /// (C) Must be chosen from 8, 16, or 32
    int32_t l=0;
    /// (F) Size of information set combinations. (Only p=1 is supported in sterncpp.)
    int32_t p=1;
    /// (F, C) Number of collision sets to check. (Ignored in the heavy Stern algorithm.)
    int32_t m=1;
    /// (C) Integer or SIMD block size in bits. Must be chosen from 8, 16, 32, or 64.
    /// Additionally 128 is available if SIMDE is enabled.
    /// By default, it is chosen as the largest available option less than 2*nclauses.
    /// If both l and m are set, it is automatically set to l*m.
    int32_t block_size=-1;
};

template<typename stern_uint_t = uint32_t>
void sterncpp(MLDProblem&, sternc_opts&);

void sterncpp_main(MLDProblem& mld_problem, sternc_opts& opts, size_t block_size);

bool sterncpp_adjust_opts(sternc_opts& opts, size_t& block_size);
#endif //STERN_STERN_H
