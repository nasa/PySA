/*
Author: Salvatore Mandra (salvatore.mandra@nasa.gov)
        Humberto Munoz-Bauza (humberto.munozbauza@nasa.gov)

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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <pysa/sat/sat.hpp>

#ifdef USE_MPI
#include <mpi.h>
#endif

int main(int argc, char* argv[]) {
  // Initialize MPI
#ifdef USE_MPI
  int provided;
  MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
  assert(MPI_THREAD_MULTIPLE == provided);

  // Duplicate worlds
  MPI_Comm mpi_comm_world;
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_world);

  // Get MPI rank and size
  const auto [mpi_rank_, mpi_size_] = []() {
    int mpi_rank_, mpi_size_;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
    return std::tuple{static_cast<std::size_t>(mpi_rank_),
                      static_cast<std::size_t>(mpi_size_)};
  }();

  // Print number of nodes
  if (mpi_rank_ == 0)
    std::cerr << "# Number of MPI nodes: " << mpi_size_ << std::endl;
#endif

  // Print the required arguments
  if (argc < 2 || argc > 6) {
    std::cerr << "Usage: " << std::filesystem::path(argv[0]).filename().string()
              << " cnf_file [max_unsat = 0] [n_threads = 0] [verbose = 0]"
              << std::endl;
    std::cerr << "Solve SAT formula in CNF format using DPLL." << std::endl
              << std::endl;
    std::cerr << "   max_unsat    Number of maximum unsatisfied clauses "
                 "that a configuration can have (default = 0)"
              << std::endl;
    std::cerr << "   n_threads    Number of threads to use (default = 0, "
                 "that is suggested by the implementation)"
              << std::endl;
    std::cerr << "   stop_on_first   Stop search when the first solution is found. "
                 "(default = 0, find all possible solutions)"
              << std::endl;
    std::cerr << "   verbose      Level of verbosity (default = 0)"
              << std::endl;
    return EXIT_FAILURE;
  }

  // Set filename for cnf formulat
  std::string cnf_file{argv[1]};

  // Set default value for maximum number of unsatisfied clasuses
  std::size_t max_unsat = 0;

  // Set default value for number of threads (0 = implementation specific)
  std::size_t n_threads = 0;

  // Default
  bool stop_on_first = false;
  // Set default value for verbosity
  std::size_t verbose = false;

  // Assign provided values
  switch (argc) {
    case 6:
      verbose = std::stoull(argv[5]);
    case 5:
      stop_on_first = std::stoull(argv[4]);
    case 4:
      n_threads = std::stoull(argv[3]);
    case 3:
      max_unsat = std::stoull(argv[2]);
  }

  // Read formula
  const auto formula = [&cnf_file]() {
    if (auto ifs = std::ifstream(cnf_file); ifs.good())
      return pysa::dpll::sat::ReadCNF(ifs);
    else
      throw std::runtime_error("Cannot open file: '" + cnf_file + "'");
  }();

  // Get results
  auto [collected, branches] =
#ifdef USE_MPI
      pysa::dpll::sat::mpi::optimize(mpi_comm_world, formula, max_unsat,
                                     verbose, n_threads);
#else
      pysa::dpll::sat::optimize(formula, max_unsat, verbose, n_threads, stop_on_first);
#endif

#ifdef USE_MPI
  if (mpi_rank_ == 0)
#endif
  {

    // sort collected states by unsatisfied clauses
    std::stable_sort(collected.begin(), collected.end(),
                     [](auto& x, auto& y) { return x.n_unsat < y.n_unsat; });

    // Print results
    for (auto& config : collected) {
      std::cerr << 'U' << config.n_unsat << " " << std::string(config.state)
                << std::endl;
    }
  }

  // Finalize MPI
#ifdef USE_MPI
  MPI_Finalize();
#endif

  // Exit
  return EXIT_SUCCESS;
}
