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
    int8_t parcheck;
    int8_t bench;
    int8_t test_hw1=false;
    int32_t nvars;
    int32_t nclauses;
    int32_t t;
    int32_t max_iters;
    int32_t l;
    int32_t p;
    int32_t m;
    int32_t block_size=-1;
};

template<typename stern_uint_t = uint32_t>
void sterncpp(MLDProblem&, sternc_opts&);

void sterncpp_main(MLDProblem& mld_problem, sternc_opts& opts, size_t block_size);

bool sterncpp_adjust_opts(sternc_opts& opts, size_t& block_size);
#endif //STERN_STERN_H
