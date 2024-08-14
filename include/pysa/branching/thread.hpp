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

#include <future>
#include <memory>
#include <thread>

namespace pysa::branching {

// Use chrono literals for time
using namespace std::chrono_literals;

// Define stop signal
using StopPtr = std::shared_ptr<bool>;
using ConstStopPtr = std::shared_ptr<const bool>;

template <typename Thread>
struct ThreadHandle {
  ThreadHandle() {
    /*
     * Empty handle.
     */
  }

  ThreadHandle(Thread &&thread, StopPtr &&stop)
      : _thread{std::move(thread)}, _stop{stop} {
    /*
     * Initialize Handle.
     */
  }

  // Delete copy constructor
  ThreadHandle(const ThreadHandle &) = delete;

  // Provide move constructor
  ThreadHandle(ThreadHandle &&thread)
      : _thread{std::move(thread._thread)}, _stop{std::move(thread._stop)} {
    /*
     * Move Ownership.
     */
  }

  // Gracefully exit
  ~ThreadHandle() {
    if (is_running()) stop();
  }

  // Delete copy assignment
  auto &operator=(const ThreadHandle &) = delete;

  // Provide move assignment
  auto &operator=(ThreadHandle &&other) {
    _thread = std::move(other._thread);
    _stop = std::move(other._stop);
    return *this;
  }

  bool is_valid() const {
    /*
     * Check if thread is valid.
     */
    return _thread.valid();
  }

  bool is_running() const {
    /*
     * Check if thread is running.
     */
    return is_valid() ? _thread.wait_for(0s) != std::future_status::ready
                      : false;
  }

  bool is_ready() const {
    /*
     * Check if thread is ready.
     */
    return is_valid() && _thread.wait_for(0s) == std::future_status::ready;
  }

  template <typename Time = std::nullptr_t>
  auto wait(Time &&time = nullptr) const {
    /*
     * Wait for 'time' or until ready.
     */

    // If not running, raise an error
    if (!is_running() and !is_ready())
      throw std::runtime_error("Thread is not running.");

    // Wait
    if constexpr (std::is_same_v<Time, std::nullptr_t>)
      return _thread.wait();
    else
      return _thread.wait_for(time);
  }

  auto get() {
    /*
     * Wait until the thread is ready, and return the results.
     */

    // If not running, raise an error
    if (!is_running() and !is_ready())
      throw std::runtime_error("Thread is not running.");

    // Get
    return _thread.get();
  }

  auto stop() {
    /*
     * Stop the thread and return results.
     */

    // If not running, raise an error
    if (!is_running() and !is_ready())
      throw std::runtime_error("Thread is not running.");

    // Set stop signal
    *_stop = true;

    // Get
    return get();
  }

 private:
  std::decay_t<Thread> _thread;
  std::decay_t<StopPtr> _stop;
};

template <typename Function, typename... Params>
auto submit(Function &&fn, Params &&...params) {
  // Initialize a stop signal
  StopPtr stop_{new bool(false)};

  // Return handle
  return ThreadHandle(
      std::async(std::forward<Function>(fn), std::forward<Params>(params)...,
                 StopPtr{stop_}),
      std::move(stop_));
}

}  // namespace pysa::branching
