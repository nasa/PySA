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

#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <vector>

#ifdef USEMPI
#include "mpi.h"
#endif
#include "libmld/mld.h"
#include "sternx/stern.h"
#include "sternx/wf.h"

using namespace boost::program_options;

extern "C" {
void sternc_(uint8_t *, sternc_opts *opts);
}

int main(int argc, char **argv) {
  // initalize mpi
  int mpi_rank = 0;
#ifdef USEMPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif
  // read program options
  std::string filenm;
  bool bench = false;
  bool use_sterncpp = false;
  bool use_hw1 = false;
  int32_t maxiters = 1000;
  int32_t max_factor = 10;
  int32_t l = -1;
  int32_t m = 1;
  int32_t p = 1;
  sternc_opts opts;
  positional_options_description posopts;
  options_description desc{"Options"};
  desc.add_options()("help,h", "Help screen")(
      "bench", bool_switch(&bench),
      "Continue until max_iters iterations and count how often the solution "
      "was found.")("sterncpp", bool_switch(&use_sterncpp),
                    "Use the optimized sterncpp implementation. Otherwise, "
                    "runs the reference Fortran implementation.")(
      "test-hw1", bool_switch(&use_hw1),
      "Check for Hamming weight <= 1 in collision bits")(
      "max_iters", value<int32_t>(&maxiters),
      "Maximum number of iterations per rank. Overrides max_factor.")(
      "block-size", value<int32_t>(&opts.block_size),
      "Block size of parity check bit columns.")(
      "col-size,l", value<int32_t>(&l), "Size of collision set to check")(
      "col-sets,m", value<int32_t>(&m), "Number of collision sets to check")(
      "combo-size,p", value<int32_t>(&p),
      "Size of information set combinations. Only p=1 is supported in "
      "sterncpp.")(
      "max_factor", value<int32_t>(&max_factor)->default_value(10),
      "Set the maximum number of iterations across all ranks to max_factor * "
      "(2**P), where P is the estimated success probability.")(
      "input", value<std::string>(&filenm), "Verbosity");

  posopts.add("input", 1);
  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(posopts).run(),
        vm);
  notify(vm);
  if (vm.count("help")) {
    if (mpi_rank == 0)
      std::cout << desc << std::endl;
#ifdef USEMPI
    MPI_Finalize();
#endif
    return 0;
  }
  if (!vm.count("input")) {
    if (mpi_rank == 0)
      std::cout << "[input] is required." << std::endl;
#ifdef USEMPI
    MPI_Finalize();
#endif
    return 1;
  }

  MLDProblem problem;

  std::ifstream ifs(filenm);
  try {
    problem.read_problem(ifs);
  } catch (MLDException &e) {
  }
  int32_t n = problem.CodeLength();
  int32_t k = problem.CodeDim();
  int32_t t = problem.Weight();

  opts.parcheck = (problem.is_parcheck() ? 1 : 0);
  opts.nvars = problem.NVars();
  opts.nclauses = problem.NClauses();
  opts.t = t;
  opts.max_iters = maxiters;
  opts.l = l;
  opts.p = p;
  opts.m = m;
  opts.bench = (bench ? 1 : 0);
  if (use_hw1)
    opts.test_hw1 = 1;

  std::vector<double> ir_dist = combo_dist(n, k, t);
  if (mpi_rank == 0) {
    std::cout << " n = " << n << '\n'
              << " k = " << k << '\n'
              << " t = " << t << '\n';
  }

  sternc_opts valid_opts;
  try {
    valid_opts = sterncpp_adjust_opts(opts);
  } catch(std::exception& e) {
#ifdef USEMPI
    MPI_Finalize();
#endif
    std::cerr << e.what();
    return 1;
  }

  double success_bits, psucc, iterwork, nts;

  if (l > 0) {
    if (mpi_rank == 0) {
      std::cout << "-- Stern --\n";
      std::cout << "l =  " << l << " | p = " << p << " | m = " << m << "\n";
    }
    success_bits = stern_nll(n, k, t, p, l, m);
    iterwork = heavy_stern_iterwf(n, k, p, l);
  } else {
    if (mpi_rank == 0)
      std::cout << "-- Heavy Stern --\n";
    success_bits = heavy_stern_nll(n, k, t, p);
    iterwork = heavy_stern_iterwf(n, k, p);
  }
  psucc = std::exp2(-success_bits);
  nts = std::log2(std::log1p(-0.99) / std::log1p(-psucc));
  if (mpi_rank == 0) {
    std::cout << "Estimated computational work:\n";
    std::cout << "(1a) Success probability per iteration : " << psucc << "\n";
    std::cout << "(1b) -lg psuccess  : " << success_bits << " bits\n";
    std::cout << "(2) Work per iteration : " << iterwork << " bits\n";
    std::cout << "(3) Expected work to success = (1b)+(2) : "
              << iterwork + success_bits << " bits" << std::endl;
    std::cout << "(4) lg NTS(99%) : " << nts << " bits\n";
    std::cout << "(5) lg TTS(99%) = (2)+(4) : " << nts + iterwork
              << " bits\n\n";
    std::cout << "Running Stern solver ... \n" << std::endl;
  }

  auto binmat = problem.clauses_as_binmat(true);
  if (use_sterncpp) {
    sterncpp_main(problem, valid_opts);
  } else {
    sternc_(binmat.data(), &valid_opts);
  }

#ifdef USEMPI
  MPI_Finalize();
  return 0;
#endif
}
