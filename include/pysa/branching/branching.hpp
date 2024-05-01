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

#include <functional>
#include <numeric>
#include <optional>
#include <thread>
#include <vector>

#include "../archive/archive.hpp"
#include "thread.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifndef NDEBUG
#include <iostream>
#endif

#ifndef NDEBUG
#include <iostream>
#endif

namespace pysa::branching {

/**
 * @brief Split a collection of branches into two.
 * Branches is a container that supports efficient .front() and .pop_front()
 * */
template <typename Branches, typename Branches_ = std::decay_t<Branches>>
auto split(Branches &&branches) {
  /*
   * Split in two by distributing branches.
   */
  std::tuple<Branches_, Branches_> v_branches_;

  // Remove first from branches and return it
  static constexpr auto pop_ = [](auto &&branches) {
    auto branch_ = branches.front();
    branches.pop_front();
    return branch_;
  };

  // Distribute branches
  for (std::size_t i_ = 0, end_ = std::size(branches); i_ < end_; ++i_) {
    auto &br_ = i_ % 2 ? std::get<0>(v_branches_) : std::get<1>(v_branches_);
    br_.push_back(pop_(branches));
  }

  // Return branches
  return v_branches_;
}

/**
 * @brief Generic braching implementation.
 * Initializes branching threads with the core procedure fn and monitors the
 * number of branches. If the branches are depleted in one thread, it is
 * rebalanced with the thread with the greatest number of branches.
 * */
template <typename Function, typename Branches, typename Time,
          template <typename...> typename Vector = std::vector>
void branching_impl(const Function &fn, Branches &branches,
                    const std::size_t n_threads, const Time &sleep_time,
                    ConstStopPtr stop) {
  // Single-threaded version
  if (n_threads == 1) {
#ifndef NDEBUG
    std::cerr << "# Initialize single-threaded pysa::branching" << std::endl;
#endif

    // Run function
    fn(branches[0], stop);

    // Multi-threaded version
  } else {
#ifndef NDEBUG
    std::cerr << "# Initialize multi-threaded pysa::branching" << std::endl;
#endif

    // Define core
    auto core_ = [fn = fn, &branches](std::size_t idx, auto &&stop) {
      fn(branches[idx], stop);
    };

    // Initialize threads
    Vector<decltype(submit(core_, 0))> threads_(n_threads);

    // Start threads
    for (std::size_t i_ = 0; i_ < n_threads; ++i_)
      threads_[i_] = submit(core_, i_);

    // Get total number of branches
    auto count_n_branches_ = [&branches]() {
      return std::transform_reduce(std::begin(branches), std::end(branches), 0,
                                   std::plus<>(),
                                   [](auto &b) { return std::size(b); });
    };

    // Get indexes to balance
    auto balance_indexes_ = [size_ = std::size(branches), &branches]() {
      // Initialize indexes for thread with the smallest/largest number of
      // branches
      std::optional<std::size_t> min_, max_;
      auto n_max_ = std::numeric_limits<std::size_t>::min();

      // Get index for threads with no branches and largest amount of branches
      for (std::size_t i_ = 0; i_ < size_; ++i_)
        if (const auto n_ = std::size(branches[i_]); !n_ && !min_)
          min_ = i_;
        else if (n_ > n_max_) {
          max_ = i_;
          n_max_ = n_;
        }

      // Return results
      return std::tuple{min_, max_};
    };

    // Avoid a race condition where count_n_branches_() may be 0 temporarily at
    // start
    std::this_thread::sleep_for(sleep_time);

    // Keep going if there are still branches or the stop signal is off
    while (count_n_branches_() && !*stop) {
      // Propagate branches between two threads
      if (const auto [ei_, ni_] = balance_indexes_(); ei_ && ni_) {
        const auto e_idx_ = ei_.value();
        const auto n_idx_ = ni_.value();

        // Stop threads
        if (threads_[e_idx_].is_running()) threads_[e_idx_].stop();
        if (threads_[n_idx_].is_running()) threads_[n_idx_].stop();

        // Get corresponding branches
        auto &n_br_ = branches[n_idx_];
        auto &e_br_ = branches[e_idx_];

        // Merge
        n_br_.splice(std::end(n_br_), e_br_);

        // Split in half
        std::tie(n_br_, e_br_) = split(n_br_);

        // Start threads
        threads_[e_idx_] = submit(core_, e_idx_);
        threads_[n_idx_] = submit(core_, n_idx_);
      }

      // Wait for a bit
      std::this_thread::sleep_for(sleep_time);
    }
  }
}

template <typename Branches,
          template <typename...> typename Vector = std::vector>
struct Brancher : ThreadHandle<std::future<void>> {
  Vector<std::decay_t<Branches>> branches;

  template <typename Function, typename Time>
  Brancher(Function &&fn, Branches &&branches, std::size_t n_threads,
           Time &&sleep_time)
      : branches(n_threads), _core{[fn, n_threads, sleep_time, this]() {
          return submit(
              [&fn, &n_threads, &sleep_time, this](ConstStopPtr stop) {
                branching_impl(fn, this->branches, n_threads, sleep_time, stop);
              });
        }} {
    /*
     * Initialize Brancher.
     */

    // Check number of threads
    if (n_threads == 0)
      throw std::runtime_error("'n_threads' must be a positive number");

    // Initialize branches
    if constexpr (std::is_lvalue_reference_v<Branches>)
      this->branches[0] = branches;
    else
      this->branches[0] = std::move(branches);
  }

  // Delete copy constructor
  Brancher(const Brancher &) = delete;

  // Add default move constructor
  Brancher(Brancher &&brancher) = default;

  // Delete copy assignment
  Brancher &operator=(const Brancher &) = delete;

  // Add default move assignment
  Brancher &operator=(Brancher &&brancher) = default;

  auto start() {
    /*
     * Start the branching.
     */

    // If already running, raise an error
    if (is_running()) throw std::runtime_error("Thread is still running.");

    // Get new thread and move ownership
    *static_cast<ThreadHandle *>(this) = _core();
  }

  auto n_branches() const {
    /*
     * Return number of remaning branches for each thread.
     */
    Vector<std::size_t> n_;
    std::transform(std::begin(branches), std::end(branches),
                   std::back_inserter(n_),
                   [](auto &&br) { return std::size(br); });
    return n_;
  }

  // Get total number of remaining branches
  auto n_total_branches() const {
    /*
     * Return total number of remaining branches.
     */
    std::size_t c_{0};
    for (const auto &n_ : n_branches()) c_ += n_;
    return c_;
  }

 private:
  const std::function<ThreadHandle(void)> _core;
};

template <typename Function, typename Branches, typename Time = decltype(1ms)>
auto branching(Function &&fn, Branches &&branches, std::size_t n_threads = 0,
               Time &&sleep_time = 1ms) {
  /*
   * Return the actual Brancher.
   */

  // If default, use 4 times the number of available threads
  if (n_threads == 0) n_threads = 4 * std::thread::hardware_concurrency();

#ifndef NDEBUG
  std::cerr << "# Using " << n_threads << " threads." << std::endl;
#endif

  // Return brancher
  return Brancher(std::forward<Function>(fn), std::forward<Branches>(branches),
                  n_threads, std::forward<Time>(sleep_time));
}

#ifdef USE_MPI
namespace mpi {

template <typename Tuple, typename Tuple_ = std::decay_t<Tuple>,
          std::size_t... I>
auto forward_tuple(Tuple &&tuple, std::index_sequence<I...>) {
  return std::tuple{
      std::forward<decltype(std::get<I>(tuple))>(std::get<I>(tuple))...};
}

template <typename Tuple>
auto forward_tuple(Tuple &&tuple) {
  return forward_tuple(
      tuple,
      std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>());
}

template <typename Buffer, typename MPI_Comm, typename T,
          typename Params = std::tuple<>>
void mpi_send_buffer(MPI_Comm &&mpi_comm_world, T &&x, std::size_t to,
                     std::size_t tag = 0, Params &&params = std::tuple{}) {
  // Get buffer
  const auto buffer_ = std::apply(
      [](auto &&...x) { return dump<Buffer>(std::forward<decltype(x)>(x)...); },
      std::tuple_cat(std::forward_as_tuple(x), forward_tuple(params)));

  // Get size of the buffer
  const std::size_t size_ = std::size(buffer_);

  // Send size of the buffer
  MPI_Send(&size_, sizeof(std::size_t), MPI_BYTE, to, tag, mpi_comm_world);

  // Send buffer
  MPI_Send(buffer_.data(), size_ * sizeof(typename Buffer::value_type),
           MPI_BYTE, to, tag, mpi_comm_world);
}

template <typename T, typename Buffer, typename MPI_Comm,
          typename Params = std::tuple<>>
auto mpi_recv_buffer(MPI_Comm &&mpi_comm_world, std::size_t from,
                     std::size_t tag = 0, Params &&params = std::tuple{}) {
  // Initialize buffer size
  std::size_t size_;

  // Receive buffer size
  MPI_Recv(&size_, sizeof(std::size_t), MPI_BYTE, from, tag, mpi_comm_world,
           MPI_STATUS_IGNORE);

  // Initialize buffer
  Buffer buffer_(size_);

  // Receive buffer
  MPI_Recv(buffer_.data(), size_ * sizeof(typename Buffer::value_type),
           MPI_BYTE, from, tag, mpi_comm_world, MPI_STATUS_IGNORE);

  // Load from buffer
  auto [head_, obj_] = std::apply(
      [](auto &&...x) {
        return load<T, Buffer>(std::forward<decltype(x)>(x)...);
      },
      std::tuple_cat(std::forward_as_tuple(std::begin(buffer_)),
                     forward_tuple(params)));

  // Check head is empty
  assert(head_ == std::end(buffer_));

  // Return object
  return obj_;
}

template <
    typename Buffer, typename Brancher, typename Time, typename MPI_Comm,
    typename DumpParams, typename LoadParams,
    typename Branches = typename decltype(std::declval<std::decay_t<Brancher>>()
                                              .branches)::value_type>
auto branching_impl(Brancher &&brancher, const Time &sleep_time,
                    MPI_Comm &&mpi_comm_world, DumpParams &&dump_params,
                    LoadParams &&load_params, ConstStopPtr stop) {
  // Get MPI rank and size
  const auto [mpi_rank_, mpi_size_] = [&mpi_comm_world]() {
    int mpi_rank_, mpi_size_;
    MPI_Comm_rank(mpi_comm_world, &mpi_rank_);
    MPI_Comm_size(mpi_comm_world, &mpi_size_);
    return std::tuple{static_cast<std::size_t>(mpi_rank_),
                      static_cast<std::size_t>(mpi_size_)};
  }();

  // Get number of branches
  auto get_n_branches_ = [&brancher, mpi_size = mpi_size_, &mpi_comm_world]() {
    // Initialize number of branches
    std::vector<std::size_t> n_(mpi_size);

    // Get number of branches
    const std::size_t nb_ = brancher.n_total_branches();

    // Collect all
    MPI_Allgather(&nb_, sizeof(nb_), MPI_BYTE, n_.data(), sizeof(nb_), MPI_BYTE,
                  mpi_comm_world);

    // Return total number of branches
    return n_;
  };

  // Get total number of branches
  auto count_n_branches_ = [get_n_branches_]() {
    const auto n_branches_ = get_n_branches_();
    return std::accumulate(std::begin(n_branches_), std::end(n_branches_), 0);
  };

  // Get indexes to balance
  auto balance_indexes_ = [&get_n_branches_]() {
    // Initialize indexes for nodes with the smallest/largest number of
    // branches
    std::optional<std::size_t> min_, max_;
    auto n_max_ = std::numeric_limits<std::size_t>::min();
    auto n_branches_ = get_n_branches_();

    // Get index for threads with no branches and largest amount of branches
    for (std::size_t i_ = 0, size_ = std::size(n_branches_); i_ < size_; ++i_)
      if (const auto n_ = n_branches_[i_]; !n_ && !min_)
        min_ = i_;
      else if (n_ > n_max_) {
        max_ = i_;
        n_max_ = n_;
      }

    // Return results
    return std::tuple{min_, max_};
  };

  // Check stop
  auto stop_ = [&mpi_comm_world, &stop]() {
    // Convert stop to int
    int s_ = *stop;
    int r_;

    // Collect all
    MPI_Allreduce(&s_, &r_, 1, MPI_INT, MPI_BAND, mpi_comm_world);

    // Return result
    return r_;
  };

  // If branches is not running, start it
  if (!brancher.is_running()) brancher.start();

  while (count_n_branches_() && !stop_()) {
    // Propagate branches between two branchers
    if (const auto [ei_, ni_] = balance_indexes_(); ei_ && ni_) {
      const auto e_idx_ = ei_.value();
      const auto n_idx_ = ni_.value();

      // For the involved MPI nodes ...
      if (mpi_rank_ == e_idx_ || mpi_rank_ == n_idx_) {
        // Stop branchers
        if (brancher.is_running()) brancher.stop();

        // Split branches
        if (mpi_rank_ == n_idx_) {
          // Merge all branches to bn_
          auto &branches_ = brancher.branches;
          auto &bn_ = branches_[0];
          for (std::size_t i_ = 1, end_ = std::size(branches_); i_ < end_; ++i_)
            bn_.splice(std::end(bn_), branches_[i_]);

          // Split in half
          std::decay_t<decltype(bn_)> be_;
          std::tie(bn_, be_) = split(bn_);

          // Send
          mpi_send_buffer<Buffer>(mpi_comm_world, be_, e_idx_, 123,
                                  dump_params);

          // Get branches
        } else
          brancher.branches[0].splice(
              std::end(brancher.branches[0]),
              mpi_recv_buffer<Branches, Buffer>(mpi_comm_world, n_idx_, 123,
                                                load_params));

        // Start again
        brancher.start();
      }
    }

    // Wait for a bit
    std::this_thread::sleep_for(sleep_time);

    // All wait here
    MPI_Barrier(mpi_comm_world);
  }

  // All wait here
  MPI_Barrier(mpi_comm_world);
}

template <typename Buffer = Buffer<>, typename MPI_Comm, typename Brancher,
          typename DumpParams = std::tuple<>,
          typename LoadParams = std::tuple<>, typename Time = decltype(60s)>
auto branching(MPI_Comm mpi_comm_world, Brancher &brancher,
               DumpParams &&dump_params = std::tuple{},
               LoadParams &&load_params = std::tuple{},
               Time &&sleep_time = 60s) {
  /*
   * Return handle to MPI brancher.
   */

  return submit([mpi_comm_world, &brancher, dump_params, load_params,
                 sleep_time](auto &&stop) {
    mpi::branching_impl<Buffer>(brancher, sleep_time, mpi_comm_world,
                                dump_params, load_params, stop);
  });
}

}  // namespace mpi
#endif

}  // namespace pysa::branching
