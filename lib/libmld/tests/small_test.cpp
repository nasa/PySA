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

#include "libmld/gauselm.h"
#include "libmld/isd.h"
#include "libmld/mld.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

typedef uint64_t my_int_t;

int main() {
  MLDProblem problem;

  // std::ifstream fs("small_test.mld");
  std::ifstream fs("test_n120_t8_k64_mld.txt");
  problem.read_problem(fs);
  //    BitMatrix<my_int_t> mat = problem.clauses_as_bitmatrix<my_int_t>();
  //    std::cout << mat << std::endl;
  uint64_t n = problem.NVars();
  uint64_t k = problem.CodeDim();
  uint32_t w = problem.Weight();
  BinMatC mat1 = problem.clauses_as_binmat(true);
  std::cout << mat1 << std::endl;
  BitMatrix<my_int_t> mat2 = problem.clauses_as_bitmatrix<my_int_t>(false);
  BitMatrix<my_int_t> Hy = problem.clauses_as_bitmatrix<my_int_t>(true);

  gaussian_elimination(Hy);
  std::cout << "Hy =\n" << Hy << std::endl;
  stern_rng_t rng{0xDEADBEEF};
  SternP1<my_int_t> stern_p1(Hy, n, k, rng);
  int counts = 0;
#ifdef NDEBUG
  int iters = 100000;
#else
  int iters = 1000;
#endif
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    stern_p1.sample_ir_split();
    stern_p1.shuffle_isd();
    bool res = stern_p1.collision_iteration(w);
    if (res)
      counts += 1;
  };
  auto end = std::chrono::steady_clock::now();
  auto dt{end - start};
  auto dtms = std::chrono::duration_cast<std::chrono::milliseconds>(dt);

  std::cout << stern_p1.get_solution_vec().as_u8_slice() << std::endl;
  std::cout << "P(success) = " << counts << " / " << iters << std::endl;
  std::cout << "t = " << dtms.count() << " ms" << std::endl;
}
