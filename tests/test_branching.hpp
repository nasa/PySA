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

#pragma once

#include <cassert>
#include <pysa/archive/archive.hpp>
#include <pysa/dpll/dpll.hpp>
#include <iostream>
#include <list>
#include <mutex>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace pysa::branching {

template <typename Branch>
auto CheckBranch(Branch &&branch) {
  return (branch.state && (branch.state % 3 == 0) && (branch.state % 31 == 0) &&
          (branch.state % 103 == 0));
}

struct Branch {
  // Full size of state
  std::size_t n;
  // State
  std::size_t state;
  // Position
  std::size_t pos;

  // Get new branches
  auto branch() const {
    if (const auto s_ = std::size_t{1} << pos; !(state & s_))
      return std::list<Branch>{Branch{n, state ^ s_, pos}};
    else
      return std::list<Branch>{};
  }

  // Update this branch
  void next() { ++pos; }

  // Check if branch is a leaf
  bool leaf() const { return pos == n; }
};

using Branches = std::list<Branch>;

void TestBranching(const std::size_t n, const std::size_t n_threads = 0,
                   const bool verbose = false) {
  // How to collect the branches
  std::mutex mutex_;
  std::vector<std::size_t> collected_;
  //
  auto collect_ = [&mutex_, &collected_](auto &&branch) {
    if (CheckBranch(branch)) {
      const std::scoped_lock<std::mutex> lock_(mutex_);
      collected_.push_back(branch.state);
    }
  };

  // Get branches
  auto brancher_ = DPLL(Branches{Branch{n, 0, 0}}, collect_, n_threads);

  // Start brancher
  auto it_ = std::chrono::high_resolution_clock::now();
  brancher_.start();

  if (verbose)
    while (brancher_.wait(1s) != std::future_status::ready)
      std::cerr << "# Number of remaining branches: "
                << brancher_.n_total_branches() << std::endl;

  // Complete branches
  brancher_.get();
  auto et_ = std::chrono::high_resolution_clock::now();

  if (verbose)
    std::cerr << "# Branchig time (ms): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(et_ -
                                                                       it_)
                     .count()
              << std::endl;

  // Sort collected numbers
  std::sort(std::begin(collected_), std::end(collected_));

  // Get head
  auto head_ = std::cbegin(collected_);

  // Check results
  for (std::size_t i_ = 0, end_ = std::size_t{1} << n; i_ < end_; ++i_)
    if (CheckBranch(Branch{n, i_, 0})) assert(*head_++ == i_);

  // All numbers should have been checked at this point
  assert(head_ == std::cend(collected_));
}

#ifdef USE_MPI
void TestBranchingMPI(const std::size_t n, const bool verbose = false) {
  // Duplicate worlds
  MPI_Comm mpi_comm_world_1;
  MPI_Comm mpi_comm_world_2;
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_world_1);
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_world_2);

  // Get MPI rank and size
  const auto [mpi_rank_, mpi_size_] = []() {
    int mpi_rank_, mpi_size_;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
    return std::tuple{static_cast<std::size_t>(mpi_rank_),
                      static_cast<std::size_t>(mpi_size_)};
  }();

  // How to collect the branches
  std::mutex mutex_;
  std::vector<std::size_t> collected_;
  //
  auto collect_ = [&mutex_, &collected_](auto &&branch) {
    if (CheckBranch(branch)) {
      const std::scoped_lock<std::mutex> lock_(mutex_);
      collected_.push_back(branch.state);
    }
  };

  // Get brancher
  auto brancher_ = [mpi_rank_ = mpi_rank_, n, &collect_]() {
    if (mpi_rank_ == 0)
      return DPLL(Branches{Branch{n, 0, 0}}, collect_);
    else
      return DPLL(Branches{}, collect_);
  }();

  // Start brancher and start MPI sync
  auto it_ = std::chrono::high_resolution_clock::now();
  auto handle_ = mpi::branching(mpi_comm_world_1, brancher_, std::tuple{},
                                std::tuple{}, 1s);

  if (verbose) {
    int n_branches_;
    while (handle_.wait(1s) != std::future_status::ready) {
      // Get number of remaining branches
      const int nb_ = brancher_.n_total_branches();
      MPI_Reduce(&nb_, &n_branches_, 1, MPI_INT, MPI_SUM, 0, mpi_comm_world_2);

      // Print number of remaining branches
      if (mpi_rank_ == 0)
        std::cerr << "# (MPI) Number of remaining branches: " << n_branches_
                  << std::endl;
    }
  }

  // Complete branching
  handle_.get();
  auto et_ = std::chrono::high_resolution_clock::now();

  if (mpi_rank_ == 0 && verbose)
    std::cerr << "# Branchig time (ms): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(et_ -
                                                                       it_)
                     .count()
              << std::endl;

  // Wait for all to finish
  MPI_Barrier(mpi_comm_world_2);

  // Collect results
  {
    static constexpr std::size_t block_size =
        sizeof(decltype(collected_)::value_type);

    // Get sizes of collected_
    std::vector<int> sizes_(mpi_size_);
    const int size_ = std::size(collected_);

    // Get sizes
    MPI_Gather(&size_, 1, MPI_INT, sizes_.data(), 1, MPI_INT, 0,
               mpi_comm_world_2);

    // Rescale to bytes
    for (auto &s_ : sizes_) s_ *= block_size;

    // Get displacement
    std::vector<int> displs_(mpi_size_, 0);
    for (std::size_t i_ = 1; i_ < mpi_size_; ++i_)
      displs_[i_] = displs_[i_ - 1] + sizes_[i_ - 1];

    // Get buffer
    decltype(collected_) all_collected_(
        mpi_rank_
            ? 0
            : (sizes_[mpi_size_ - 1] + displs_[mpi_size_ - 1]) / block_size);

    // Gather all
    MPI_Gatherv(collected_.data(), size_ * block_size, MPI_BYTE,
                all_collected_.data(), sizes_.data(), displs_.data(), MPI_BYTE,
                0, mpi_comm_world_2);

    // Move to collected_
    collected_ = std::move(all_collected_);
  }

  if (mpi_rank_ == 0) {
    // Sort collected numbers
    std::sort(std::begin(collected_), std::end(collected_));

    // Get head
    auto head_ = std::cbegin(collected_);

    // Check results
    for (std::size_t i_ = 1, end_ = std::size_t{1} << n; i_ < end_; ++i_)
      if (CheckBranch(Branch{n, i_, 0})) assert(*head_++ == i_);

    // All numbers should have been checked at this point
    assert(head_ == std::cend(collected_));
  }

  // Free MPI_Comm
  MPI_Comm_free(&mpi_comm_world_1);
  MPI_Comm_free(&mpi_comm_world_2);
}
#endif

}  // namespace pysa::branching
