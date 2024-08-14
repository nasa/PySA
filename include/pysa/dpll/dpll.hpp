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

#include "../branching/branching.hpp"

namespace pysa::branching {

template <bool depth_first = true, bool exit_on_first = false, typename Branches, typename Collect>
void DPLL_(Branches &&branches, Collect &&collect, StopPtr stop) {
  // While there are still branches ...
  while (std::size(branches) && !*stop) {
    // Get last branch (depth first)
    auto branch_ = [&branches]() {
      if constexpr (depth_first) {
        auto branch_ = std::move(branches.back());
        branches.pop_back();
        return branch_;
      } else {
        auto branch_ = std::move(branches.front());
        branches.pop_front();
        return branch_;
      }
    }();

    // Depth-first
    for (; !branch_.leaf(); branch_.next())
      branches.splice(std::end(branches), branch_.branch());

    // Collect
    if constexpr (exit_on_first){
      if(collect(std::move(branch_))){
        *stop = true;
        break;
      }
    } else {
      collect(std::move(branch_));
    }
  }
}

template <bool depth_first = true, bool exit_on_first = false, typename Branches, typename Collect,
          typename... Args>
auto DPLL(Branches &&branches, Collect &&collect, Args &&...args) {
  /*
   * Simplified version of `branching` to run DPLL-like branches.
   */

  // Get brancher
  return branching(
      [collect](auto &&branches, auto &&stop) {
        DPLL_<depth_first, exit_on_first>(branches, collect, stop);
      },
      std::forward<Branches>(branches), std::forward<Args>(args)...);
}

}  // namespace pysa::branching
