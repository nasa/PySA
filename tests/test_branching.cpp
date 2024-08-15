/*
Author: Salvatore Mandra (salvatore.mandra@nasa.gov)

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

#include "test_branching.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

int main() {
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

  // Print the number of nodes
  if (mpi_rank_ == 0)
    std::cerr << "# Number of MPI nodes: " << mpi_size_ << std::endl;
#endif

  // Check Branching
#ifdef USE_MPI
  if (mpi_rank_ == 0)
#endif
  {
    // Use one thread
    pysa::branching::TestBranching(28, 1, true);
    pysa::branching::TestBranching<true>(28, 1, true);
    // Use number of threads provided by the implementation
    pysa::branching::TestBranching(30, 0, true);
    pysa::branching::TestBranching<true>(30, 0, true);
  }
#ifdef USE_MPI
  MPI_Barrier(mpi_comm_world);
#endif

  // Check Branching with MPI
#ifdef USE_MPI
  pysa::branching::TestBranchingMPI(30, true);
#endif

  // Finalize
#ifdef USE_MPI
  MPI_Finalize();
#endif
}
