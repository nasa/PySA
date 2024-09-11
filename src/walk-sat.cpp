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

#include "pysa/sat/walksat.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>


using namespace std::chrono_literals;

typedef std::vector<std::tuple<std::string, size_t, unsigned>> walksat_result_t;

namespace pysa::sat{
walksat_result_t walksat_optimize_bench(
    const Formula_WS& formula, 
    uint32_t max_steps,
    double p, 
    uint32_t max_unsat, 
    uint64_t seed, 
    double timeout,
    bool bench)
  {

  int timeout_ms = timeout > 0 ? int(timeout * 1000) : 0;
  // Get initial time
  size_t nsols = 0;
  size_t nits = 0;
  size_t total_nsteps = 0;
  std::unordered_map<std::vector<uint8_t>, std::pair<size_t, unsigned long>,
                     bitvector_hash>
      sol_map;
  std::vector<size_t> nstep_hist;
  pysa::sat::WalkSatOptimizer wsopt(formula, seed, p);
  auto it_ = std::chrono::high_resolution_clock::now();
  do {
    wsopt.restart_state();
    size_t nsteps = 0;
    uint64_t n_unsat = formula.num_clauses();
    for (; nsteps < max_steps; ++nsteps) {
      n_unsat = wsopt.step();
      if (n_unsat <= max_unsat)
        break;
      // occassionally check for timeout in the inner loop
      if(timeout_ms > 0 && (nsteps+1)%1000 == 0)
        if(std::chrono::high_resolution_clock::now() - it_ >
            std::chrono::milliseconds(timeout_ms))
          break;
    }
    auto state = wsopt.state();
    if (n_unsat <= max_unsat) {
      ++nsols;
      nstep_hist.push_back(nsteps);
      sol_map[state].first += 1;
      sol_map[state].second = n_unsat;
    }
    total_nsteps += nsteps;
    nits += 1;
    if (!bench)
      break;
  } while (std::chrono::high_resolution_clock::now() - it_ <
           std::chrono::milliseconds(int(timeout * 1000)));
  // Get final time
  auto et_ = std::chrono::high_resolution_clock::now();
  double duration_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(et_ - it_).count() /
      1000.0;
  // Sort solutions lexicographically
  std::vector<std::tuple<std::string, size_t, unsigned>> sols_vec;
  sols_vec.reserve(sol_map.size());
  for (const auto &kv : sol_map) {
    sols_vec.emplace_back(bitvector_string(kv.first), kv.second.first,
                          kv.second.second);
  }
  std::stable_sort(sols_vec.begin(), sols_vec.end(), [](auto &x, auto &y) {
    return std::get<0>(x) < std::get<0>(y);
  });
  std::stable_sort(sols_vec.begin(), sols_vec.end(), [](auto &x, auto &y) {
    return std::get<2>(x) < std::get<2>(y);
  });
  // Sort distribution of nsteps
  std::sort(nstep_hist.begin(), nstep_hist.end());
  std::cout << "C Solution count = " << nsols << "\n"
            << "C Unique solutions found = " << sol_map.size() << "\n"
            << "C Computation time (ms) = " << duration_ms << "\n"
            << "C Number of restarts = " << nits << "\n"
            << "C Total inner steps = " << total_nsteps << "\n"
            << "C Avg steps to solution = " << double(total_nsteps) / nsols
            << "\n"
            << "C Avg time per step (ms) = "
            << double(duration_ms) / total_nsteps << "\n";

  double qlist[4]{0.5, 0.9, 0.95, 0.99};
  for (double q : qlist)
    if (nits >= 10) {
      double qn = double(nstep_hist.size()) * q;
      size_t i1 = (size_t)std::floor(qn) - 1;
      size_t i2 = (size_t)std::ceil(qn) - 1;
      double ip = qn - 1.0 - double(i1);
      double qnt = (1.0 - ip) * nstep_hist[i1] + ip * nstep_hist[i2];
      std::cout << "C NTS(" << q * 100 << "%) = " << qnt << "\n";
    }
  std::cout << std::endl;
  for (const auto &kv : sols_vec) {
    std::cout << "U" << std::get<2>(kv) << " " << std::get<0>(kv) << "  "
              << std::get<1>(kv) << "\n";
  }
  std::cout << std::endl;

  return sols_vec;
}
}