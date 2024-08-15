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

#include <cstddef>
#include <iostream>
#include <mutex>
#include <pysa/dpll/dpll.hpp>
#include <thread>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace pysa::dpll {

using namespace std::chrono_literals;

template <bool depth_first = true, bool stop_on_first = false, typename Init, typename Get,
          typename WallTime = std::nullptr_t,
          typename SleepTime = decltype(1ms)>
auto DPLL(Init &&init, Get &&get, bool verbose = false,
          std::size_t n_threads = std::thread::hardware_concurrency(),
          WallTime &&walltime = nullptr, SleepTime &&sleep_time = 1ms) {
  // Initialize root
  auto root_ = init();

  // Get branches type
  using branches_type = std::decay_t<decltype(root_.branch())>;

  // How to collect results
  std::vector<decltype(get(root_))> collected_;

  std::mutex mutex_;
  auto collect_ = [&mutex_, &collected_, &get](auto &&branch) {
#ifndef NDEBUG
    if (!branch.leaf()) throw std::runtime_error("Not a leaf");
#endif
    if (!branch.partial()) {
      const std::scoped_lock<std::mutex> lock_(mutex_);
      collected_.push_back(get(branch));
      return true;
    } else {
      return false;
    }
  };

  // Get brancher
  auto brancher_ = pysa::branching::DPLL<depth_first, stop_on_first>(
      branches_type{{root_}}, collect_, n_threads, sleep_time);

  // Get initial time
  auto it_ = std::chrono::high_resolution_clock::now();

  // Start branching
  brancher_.start();

  // If no walltime is provided ...
  if constexpr (std::is_same_v<std::decay_t<WallTime>, std::nullptr_t>) {
    if (verbose)
      while (brancher_.wait(1s) != std::future_status::ready)
        std::cerr << "# Number of remaining branches: "
                  << brancher_.n_total_branches() << std::endl;

    // Complete branches
    brancher_.get();

    // Otherwise, run branches for the walltime amount
  } else {
    if (verbose) std::cerr << "# Start branching ... ";

    // Wait the desired amount of time
    brancher_.wait(walltime);

    // Stop branches
    brancher_.stop();

    if (verbose) std::cerr << "Done!" << std::endl;
  }

  // Get final ti,e
  auto et_ = std::chrono::high_resolution_clock::now();

  if (verbose)
    std::cerr << "# Branching time (ms): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(et_ -
                                                                       it_)
                     .count()
              << std::endl;

  // Combine branches
  branches_type branches_;
  for (auto &b_ : brancher_.branches) branches_.splice(std::end(branches_), b_);

  // Return results
  return std::pair{collected_, branches_};
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

template <typename MPI_Comm_World, typename Array,
          typename DumpParams = std::tuple<>,
          typename LoadParams = std::tuple<>>
auto _Gather(MPI_Comm_World &&mpi_comm_world, Array &&array,
             DumpParams &&dump_params = std::tuple<>{},
             LoadParams &&load_params = std::tuple<>{}) {
  // Get MPI rank and size
  const auto [mpi_rank_, mpi_size_] = [&mpi_comm_world]() {
    int mpi_rank_, mpi_size_;
    MPI_Comm_rank(mpi_comm_world, &mpi_rank_);
    MPI_Comm_size(mpi_comm_world, &mpi_size_);
    return std::tuple{static_cast<std::size_t>(mpi_rank_),
                      static_cast<std::size_t>(mpi_size_)};
  }();

  // Get buffers
  auto buffer_ = std::apply(
      [](auto &&...x) {
        return pysa::branching::dump(std::forward<decltype(x)>(x)...);
      },
      std::tuple_cat(std::forward_as_tuple(array), forward_tuple(dump_params)));

  // Get size of blocks
  static constexpr std::size_t block_size =
      sizeof(typename decltype(buffer_)::value_type);

  // Get sizes of collected_
  std::vector<int> sizes_(mpi_size_);
  const int size_ = std::size(buffer_);

  // Get sizes
  MPI_Gather(&size_, 1, MPI_INT, sizes_.data(), 1, MPI_INT, 0, mpi_comm_world);

  // Rescale to bytes
  for (auto &s_ : sizes_) s_ *= block_size;

  // Get displacement
  std::vector<int> displs_(mpi_size_, 0);
  for (std::size_t i_ = 1; i_ < mpi_size_; ++i_)
    displs_[i_] = displs_[i_ - 1] + sizes_[i_ - 1];

  // Get buffer
  std::remove_cv_t<decltype(buffer_)> all_buffer_(
      mpi_rank_
          ? 0
          : (sizes_[mpi_size_ - 1] + displs_[mpi_size_ - 1]) / block_size);

  // Gather all
  MPI_Gatherv(buffer_.data(), size_ * block_size, MPI_BYTE, all_buffer_.data(),
              sizes_.data(), displs_.data(), MPI_BYTE, 0, mpi_comm_world);

  // Clear buffer
  buffer_.clear();

  // Initialize collection
  std::decay_t<decltype(array)> collect_;

  // Initialize buffer's head
  auto head_ = std::cbegin(all_buffer_);
  while (head_ != std::cend(all_buffer_)) {
    // Load collected
    auto [thead_, b_] = std::apply(
        [](auto &&...x) {
          return pysa::branching::load<decltype(collect_)>(
              std::forward<decltype(x)>(x)...);
        },
        std::tuple_cat(std::forward_as_tuple(head_),
                       forward_tuple(load_params)));

    // Move
    collect_.insert(std::end(collect_), std::make_move_iterator(std::begin(b_)),
                    std::make_move_iterator(std::end(b_)));

    // Update head
    head_ = thead_;
  }

  // Return all collected
  return collect_;
}

template <typename MPI_Comm_World, typename Array,
          typename DumpParams = std::tuple<>,
          typename LoadParams = std::tuple<>>
auto _Scatter(MPI_Comm_World &&mpi_comm_world, Array &&array,
              DumpParams &&dump_params = std::tuple<>{},
              LoadParams &&load_params = std::tuple<>{}) {
  // Get MPI rank and size
  const auto [mpi_rank_, mpi_size_] = [&mpi_comm_world]() {
    int mpi_rank_, mpi_size_;
    MPI_Comm_rank(mpi_comm_world, &mpi_rank_);
    MPI_Comm_size(mpi_comm_world, &mpi_size_);
    return std::tuple{static_cast<std::size_t>(mpi_rank_),
                      static_cast<std::size_t>(mpi_size_)};
  }();

  // Check size
  assert(mpi_rank_ || std::size(array) == mpi_size_);

  // Initialize sizes
  std::vector<int> sizes_;

  // Initialize buffer
  decltype(pysa::branching::dump(array.front())) buffer_;

  // Get size of blocks
  static constexpr std::size_t block_size =
      sizeof(typename decltype(buffer_)::value_type);

  // Collect buffer to a single
  if (mpi_rank_ == 0)
    for (const auto &x_ : array) {
      // Get buffer
      auto tbuffer_ = std::apply(
          [](auto &&...x) {
            return pysa::branching::dump(std::forward<decltype(x)>(x)...);
          },
          std::tuple_cat(std::forward_as_tuple(x_),
                         forward_tuple(dump_params)));

      // Update size
      sizes_.push_back(block_size * std::size(tbuffer_));

      // Update buffer
      buffer_ += std::move(tbuffer_);
    }

  // Get displacement
  std::vector<int> displs_(mpi_rank_ ? 0 : mpi_size_, 0);
  if (mpi_rank_ == 0)
    for (std::size_t i_ = 1; i_ < mpi_size_; ++i_)
      displs_[i_] = displs_[i_ - 1] + sizes_[i_ - 1];

  // Initialize output
  auto sbuffer_ = [&sizes_, &mpi_comm_world]() {
    int size_;
    MPI_Scatter(sizes_.data(), 1, MPI_INT, &size_, 1, MPI_INT, 0,
                mpi_comm_world);
    return decltype(buffer_)(size_ / block_size);
  }();

  // Scatter
  MPI_Scatterv(buffer_.data(), sizes_.data(), displs_.data(), MPI_BYTE,
               sbuffer_.data(), std::size(sbuffer_) * block_size, MPI_BYTE, 0,
               mpi_comm_world);

  // Clear buffer
  buffer_.clear();

  // Load array
  auto [head_, array_] = std::apply(
      [](auto &&...x) {
        return pysa::branching::load<typename std::decay_t<Array>::value_type>(
            std::forward<decltype(x)>(x)...);
      },
      std::tuple_cat(std::forward_as_tuple(std::begin(sbuffer_)),
                     forward_tuple(load_params)));

  // Check head
  assert(head_ == std::end(sbuffer_));

  // Return array
  return array_;
}

template <typename MPI_Comm_World, typename Object,
          typename DumpParams = std::tuple<>,
          typename LoadParams = std::tuple<>>
auto _Bcast(MPI_Comm_World &&mpi_comm_world, const Object &object,
            const std::size_t root, DumpParams &&dump_params = std::tuple<>{},
            LoadParams &&load_params = std::tuple<>{}) {
  // Get MPI rank and size
  const auto [mpi_rank_, mpi_size_] = [&mpi_comm_world]() {
    int mpi_rank_, mpi_size_;
    MPI_Comm_rank(mpi_comm_world, &mpi_rank_);
    MPI_Comm_size(mpi_comm_world, &mpi_size_);
    return std::tuple{static_cast<std::size_t>(mpi_rank_),
                      static_cast<std::size_t>(mpi_size_)};
  }();

  // Initialize size
  int size_;

  // Initialize buffer
  decltype(pysa::branching::dump(object)) buffer_;

  // Get size of blocks
  static constexpr std::size_t block_size =
      sizeof(typename decltype(buffer_)::value_type);

  // Get buffer
  if (mpi_rank_ == root) {
    buffer_ = std::apply(
        [](auto &&...x) {
          return pysa::branching::dump(std::forward<decltype(x)>(x)...);
        },
        std::tuple_cat(std::forward_as_tuple(object),
                       forward_tuple(dump_params)));
    size_ = block_size * std::size(buffer_);
  }

  // Broadcast size
  MPI_Bcast(&size_, 1, MPI_INT, root, mpi_comm_world);

  // Resize
  if (mpi_rank_ != root) buffer_.resize(size_ / block_size);

  // Broadcast
  MPI_Bcast(buffer_.data(), size_, MPI_BYTE, root, mpi_comm_world);

  // Load object
  auto [head_, object_] = std::apply(
      [](auto &&...x) {
        return pysa::branching::load<Object>(std::forward<decltype(x)>(x)...);
      },
      std::tuple_cat(std::forward_as_tuple(std::begin(buffer_)),
                     forward_tuple(load_params)));

  // Check head
  assert(head_ == std::end(buffer_));

  // Return array
  return object_;
}

template <typename Init, typename Get, typename ThreadSleepTime = decltype(1ms),
          typename SleepTime = decltype(60s),
          typename GetDumpParams = std::tuple<>,
          typename GetLoadParams = std::tuple<>,
          typename BranchDumpParams = std::tuple<>,
          typename BranchLoadParams = std::tuple<>>
auto DPLL(Init &&init, Get &&get, bool verbose = false,
          std::size_t n_threads = std::thread::hardware_concurrency(),
          GetDumpParams &&get_dump_params = std::tuple{},
          GetLoadParams &&get_load_params = std::tuple{},
          BranchDumpParams &&branch_dump_params = std::tuple{},
          BranchLoadParams &&branch_load_params = std::tuple{},
          SleepTime &&sleep_time = 60s,
          ThreadSleepTime &&thread_sleep_time = 1ms) {
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

  if (verbose && mpi_rank_ == 0)
    std::cerr << "# Starting breadth-first search ... ";

  // Run a breadth-first search to fill branches
  auto [collected_, partial_branches_] = [&init, &get, mpi_rank_ = mpi_rank_,
                                          n_threads]()
      -> decltype(pysa::dpll::DPLL<false>(init, get, false, n_threads, 1s)) {
    if (mpi_rank_ == 0) {
      return pysa::dpll::DPLL<false>(init, get, false, n_threads, 1s);
    } else
      return {};
  }();

  if (verbose && mpi_rank_ == 0) std::cerr << "Done!" << std::endl;

  // Split into branches
  std::vector<decltype(partial_branches_)> v_partial_branches_(mpi_size_);
  if (mpi_rank_ == 0) {
    std::size_t i_ = 0;
    for (auto &x_ : partial_branches_)
      v_partial_branches_[i_++ % mpi_size_].push_back(std::move(x_));
  }

  // Wait
  MPI_Barrier(mpi_comm_world_2);

  std::mutex mutex_;
  auto collect_ = [&mutex_, &collected_ = collected_, &get](auto &&branch) {
#ifndef NDEBUG
    if (!branch.leaf()) throw std::runtime_error("Not a leaf");
#endif
    if (!branch.partial()) {
      const std::scoped_lock<std::mutex> lock_(mutex_);
      collected_.push_back(get(branch));
    }
  };

  // Get branches and start brancher
  auto brancher_ =
      pysa::branching::DPLL(_Scatter(mpi_comm_world_2, v_partial_branches_,
                                     branch_dump_params, branch_load_params),
                            collect_, n_threads, thread_sleep_time);

  // Start brancher and start MPI sync
  auto it_ = std::chrono::high_resolution_clock::now();

  // Get handle
  auto handle_ = pysa::branching::mpi::branching(
      mpi_comm_world_1, brancher_, branch_dump_params, branch_load_params,
      sleep_time);

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

  // Combine branches
  decltype(partial_branches_) branches_;
  for (auto &b_ : brancher_.branches) branches_.splice(std::end(branches_), b_);

  // Wait for all to finish
  MPI_Barrier(mpi_comm_world_2);

  // Collect results
  collected_ =
      _Gather(mpi_comm_world_2, collected_, get_dump_params, get_load_params);

  // Wait for all to finish
  MPI_Barrier(mpi_comm_world_2);

  // Collect branches
  branches_ = _Gather(mpi_comm_world_2, branches_, branch_dump_params,
                      branch_load_params);

  // Wait for all to finish
  MPI_Barrier(mpi_comm_world_2);

  // Free MPI_Comm
  MPI_Comm_free(&mpi_comm_world_1);
  MPI_Comm_free(&mpi_comm_world_2);

  // Return results
  return std::pair{collected_, branches_};
}

}  // namespace mpi
#endif

}  // namespace pysa::dpll
