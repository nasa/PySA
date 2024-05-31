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

#include <iostream>
#include <unordered_set>

#include "../include/pysa/sat/sat.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

using namespace pysa::dpll;
using namespace std::chrono_literals;

template <typename Formula, typename Conf>
auto CountUnsat(const Formula &formula, const Conf &conf) {
  std::size_t n_unsat_ = 0;
  for (const auto &clause_ : formula) {
    bool sat_ = false;
    for (const auto &x_ : clause_)
      sat_ |= (x_ < 0) ^ conf.test(std::abs(x_) - 1);
    n_unsat_ += !sat_;
  }
  return n_unsat_;
}

struct ConfigurationHash {
  template <typename Configuration>
  auto operator()(Configuration &&conf) const {
    return conf.hash();
  }
};

template <typename Formula>
auto ExhaustiveEnumeration(const Formula &formula,
                           std::size_t max_n_unsat = 0) {
  // Count number of variables
  const auto n_vars_ = sat::GetNVars(formula);

  // Check that number of variables fit in a std::size_t
  if (n_vars_ > 8 * sizeof(std::size_t))
    throw std::runtime_error("Number of variables is too large");

  // Check that n_vars_ is smaller than the number of bits in a `BitSet` block
  if (n_vars_ > BitSet<>::block_size)
    throw std::runtime_error("Too many variables");

  // Initialize set of configurations
  std::vector<sat::Configuration<BitSet<>, std::size_t>> collect_;

  // For each configuration, get number of unsat clauses
  for (std::size_t i_ = 0, end_ = std::size_t{1} << n_vars_; i_ < end_; ++i_) {
    auto s_ = BitSet<>(n_vars_, std::vector<BitSet<>::block_type>{{i_}});
    if (const auto n_unsat_ = CountUnsat(formula, s_); n_unsat_ <= max_n_unsat)
      collect_.emplace_back(std::move(s_), n_unsat_);
  }

  // Convert to set
  return std::unordered_set<decltype(collect_)::value_type, ConfigurationHash>(
      std::make_move_iterator(std::begin(collect_)),
      std::make_move_iterator(std::end(collect_)));
}

auto TestBitSet(const std::size_t n) {
  // How to check padding
  const auto check_padding = [n](auto &&buffer) {
    if (const auto s_ = n % BitSet<>::block_size; s_)
      return !popcount(buffer.back() & ~((BitSet<>::block_type{1} << s_) - 1));
    else
      return true;
  };

  // Initialize random engine
  std::mt19937_64 rng_(std::random_device{}());

  // Initialize BitSet
  BitSet<> bs_;
  for (std::size_t i_ = 0; i_ < n; ++i_) bs_.push_back(rng_() % 2);

  // Check padding
  assert(check_padding(bs_.buffer()));

  // Check create from buffer
  assert(decltype(bs_)(std::size(bs_), bs_.buffer()) == bs_);

  // Add random bits beyond 'n'
  if (const auto s_ = n % BitSet<>::block_size; s_) {
    // Get buffer
    auto buffer1_ = bs_.buffer();
    auto buffer2_ = bs_.buffer();

    // Check padding
    assert(check_padding(buffer1_));
    assert(check_padding(buffer2_));

    // Add random bits
    buffer1_.back() |= rng_() & ~((BitSet<>::block_type{1} << s_) - 1);
    buffer2_.back() |= rng_() & ~((BitSet<>::block_type{1} << s_) - 1);

    // Padding should be wrong
    assert(!check_padding(buffer1_));
    assert(!check_padding(buffer2_));

    // Get a new bitset
    auto tbs1_ = BitSet<>(n, buffer1_);
    auto tbs2_ = BitSet<>(n, std::move(buffer2_));

    // Padding should be ok
    assert(check_padding(tbs1_.buffer()) && tbs1_ == bs_);
    assert(check_padding(tbs2_.buffer()) && tbs2_ == bs_);
  }

  // Check load/dump
  {
    // Get buffer
    auto buffer_ = pysa::branching::dump(bs_);

    // Load
    const auto [h_, tbs_] =
        pysa::branching::load<decltype(bs_)>(std::begin(buffer_));

    // Check
    assert(h_ == std::end(buffer_) && tbs_ == bs_);
  }
}

auto TestBranch(std::optional<std::size_t> seed = std::nullopt) {
  // Get random SAT problem
  const auto formula_ = sat::GetRandomInstance(4, 20, 30, seed);

  // Get root
  sat::Branch root_{formula_, 10};

  // Check dump/load
  {
    const auto buffer_ = pysa::branching::dump(root_);
    const auto [h_, troot_] = pysa::branching::load<decltype(root_)>(
        std::begin(buffer_), root_.instance);
    assert(h_ == std::end(buffer_) && root_ == troot_);
  }
  //
  {
    auto x_ = sat::Configuration{root_.state(), root_.n_unsat()};
    const auto buffer_ = pysa::branching::dump(x_);
    const auto [h_, y_] =
        pysa::branching::load<decltype(x_)>(std::begin(buffer_));
    assert(h_ == std::end(buffer_) && x_ == y_);
  }
  //
  {
    auto x_ = std::list{sat::Configuration{root_.state(), root_.n_unsat()}};
    const auto buffer_ = pysa::branching::dump(x_);
    const auto [h_, tx_] =
        pysa::branching::load<decltype(x_)>(std::begin(buffer_));
    assert(h_ == std::end(buffer_) && x_ == tx_);
  }
  //
  {
    auto x_ = std::vector{sat::Configuration{root_.state(), root_.n_unsat()}};
    const auto buffer_ = pysa::branching::dump(x_);
    const auto [h_, tx_] =
        pysa::branching::load<decltype(x_)>(std::begin(buffer_));
    assert(h_ == std::end(buffer_) && x_ == tx_);
  }
  //
  {
    auto x_ = std::list{{root_}};
    const auto buffer_ = pysa::branching::dump(x_);
    const auto [h_, tx_] = pysa::branching::load<decltype(x_)>(
        std::begin(buffer_), root_.instance);
    assert(h_ == std::end(buffer_) && x_ == tx_);
  }
}

auto TestDPLLSAT(std::size_t k, std::size_t n, std::size_t m,
                 std::size_t max_n_unsat, bool verbose = false,
                 const std::optional<std::size_t> seed = std::nullopt) {
  // Get random SAT problem
  const auto formula_ = sat::GetRandomInstance(k, n, m, seed);

  // Get configurations from exhaustive enumeration
  if (verbose) std::cerr << "# Start exhaustive enumeration ... ";
  const auto all_ = ExhaustiveEnumeration(formula_, max_n_unsat);
  if (verbose) std::cerr << "Done!" << std::endl;

  // Get configurations from dpll
  const auto [dpll_, branches_] = sat::optimize(formula_, max_n_unsat, verbose);

  // Branches should be empty
  if (std::size(branches_))
    throw std::runtime_error("Some branches are not used");

  // Check number of unsat
  for (const auto &x_ : dpll_)
    if (x_.n_unsat != CountUnsat(formula_, x_.state))
      throw std::runtime_error("Wrong number of unsat clauses");

  // Check size
  if (std::size(dpll_) != std::size(all_))
    throw std::runtime_error("Collected wrong number of states");

  // Check if everything has been properly collected
  for (const auto &state_ : dpll_)
    if (all_.find(state_) == std::end(all_))
      throw std::runtime_error("Got a wrong state");
}

#ifdef USE_MPI
auto TestDPLLSAT_MPI(std::size_t k, std::size_t n, std::size_t m,
                     std::size_t max_n_unsat = 0, bool verbose = false,
                     std::optional<std::size_t> seed = std::nullopt) {
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

  // Get random formula
  auto formula_ = [k, n, m, mpi_rank_ = mpi_rank_,
                   seed]() -> decltype(sat::GetRandomInstance(0, 0, 0, 0)) {
    if (mpi_rank_ == 0)
      return sat::GetRandomInstance(k, n, m, seed);
    else
      return {};
  }();

  // Collect results using MPI
  auto [mpi_conf_, branches_] =
      sat::mpi::optimize(mpi_comm_world, formula_, max_n_unsat, verbose,
                         std::thread::hardware_concurrency(), 1s);

  if (mpi_rank_ == 0) {
    // Collect results and convert them to set
    const auto conf_ = [&formula_, max_n_unsat, verbose]() {
      auto [conf_, branches_] = sat::optimize(formula_, max_n_unsat, verbose);
      assert(std::size(branches_) == 0);
      return std::unordered_set<decltype(conf_)::value_type, ConfigurationHash>(
          std::make_move_iterator(std::begin(conf_)),
          std::make_move_iterator(std::end(conf_)));
    }();

    // Check no branches are left
    assert(std::size(branches_) == 0);

    // Check size
    assert(std::size(conf_) == std::size(mpi_conf_));

    // Check states
    for (const auto &s_ : mpi_conf_) assert(conf_.find(s_) != std::end(conf_));
  }

  // Wait for all
  MPI_Barrier(mpi_comm_world);

  // Free MPI_Comm
  MPI_Comm_free(&mpi_comm_world);
}
#endif
