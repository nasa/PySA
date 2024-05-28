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

#include <optional>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "branch.hpp"
#include "cnf.hpp"
#include "instance.hpp"

namespace pysa::dpll::sat {

/**
 * @brief Main routine to optimize a SAT formula.
 * @tparam Formula SAT Formula type. See Instance(const & Formula) constructor.
 * @tparam WallTime
 * @tparam SleepTime
 * @param formula Object of type Formula
 * @param max_n_unsat Maximum number of unsatisfiable clauses allowed
 * @param verbose Print verbose information to cerr
 * @param n_threads Number of concurrent threads
 * @param walltime
 * @param sleep_time
 * @return
 */
template <typename Formula, typename WallTime = std::nullptr_t,
          typename SleepTime = decltype(1ms)>
auto optimize(Formula &&formula, std::size_t max_n_unsat = 0,
              bool verbose = false,
              std::optional<std::size_t> n_threads = std::nullopt,
              WallTime &&walltime = nullptr, SleepTime &&sleep_time = 1ms) {
  // Get root initializer
  const auto init_ = [&formula, max_n_unsat]() {
    return Branch<>(formula, max_n_unsat);
  };

  // How to collect results from branch
  const auto get_ = [](auto &&branch) {
    return Configuration{branch.state(), branch.n_unsat()};
  };

  // Get configurations from dpll
  return DPLL(init_, get_, verbose,
              n_threads.value_or(std::thread::hardware_concurrency()), walltime,
              sleep_time);
}

#ifdef USE_MPI
namespace mpi {

template <typename MPI_Comm_World, typename Formula,
          typename SleepTime = decltype(60s),
          typename ThreadSleepTime = decltype(1ms)>
auto optimize(MPI_Comm_World &&mpi_comm_world, Formula formula,
              std::size_t max_n_unsat, bool verbose = false,
              std::optional<std::size_t> n_threads = std::nullopt,
              SleepTime &&sleep_time = 60s,
              ThreadSleepTime &&thread_sleep_time = 1ms) {
  // Broadcast it
  {
    // Dump to buffer
    auto buffer_ = pysa::branching::dump(formula);
    static constexpr auto block_size_ =
        sizeof(typename decltype(buffer_)::value_type);

    // Brodcast size
    int size_ = std::size(buffer_);
    MPI_Bcast(&size_, 1, MPI_INT, 0, mpi_comm_world);

    // Broadcast buffer
    buffer_.resize(size_);
    MPI_Bcast(buffer_.data(), block_size_ * size_, MPI_BYTE, 0, mpi_comm_world);

    // Dump to formula
    decltype(std::cbegin(buffer_)) head_;
    std::tie(head_, formula) =
        pysa::branching::load<decltype(formula)>(std::begin(buffer_));
  }

  // Get instance
  const auto instance_ = std::make_shared<sat::Instance<>>(formula);

  // How to initialize root
  const auto init_ = [&instance_, max_n_unsat]() {
    return Branch<>(instance_, max_n_unsat);
  };

  // How to collect results from branch
  const auto get_ = [](auto &&branch) {
    return Configuration{branch.state(), branch.n_unsat()};
  };

  // Get configurations from dpll
  return pysa::dpll::mpi::DPLL(
      init_, get_, verbose,
      n_threads.value_or(std::thread::hardware_concurrency()), {}, {}, {},
      std::tuple{instance_}, sleep_time, thread_sleep_time);
}

} // namespace mpi
#endif

} // namespace pysa::dpll::sat
