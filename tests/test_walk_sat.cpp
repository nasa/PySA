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

#include <iostream>
#include <optional>
#include <random>
#include <unordered_set>

#include "pysa/sat/walksat.hpp"



pysa::sat::Formula_WS GetRandomInstance(const std::size_t k, const std::size_t n,
                       const std::size_t m,
                       const std::optional<std::size_t> seed = std::nullopt) {
  /*
   * Generate k-SAT random instance.
   */
  using clause_type = pysa::sat::Clause_WS;

  // Check
  if (n < k)
    throw std::runtime_error("'n' must be larger than 'k'");

  // Get random seed
  const std::size_t seed_ = seed.value_or(std::random_device()());

  // Initialize random generator
  std::mt19937_64 rng_(seed_);

#ifndef NDEBUG
  std::cerr << "# Used seed: " << seed_ << std::endl;
#endif

  // Initialize set of indexes
  std::vector<int> indexes_(n);
  for (std::size_t i_ = 0; i_ < n; ++i_)
    indexes_[i_] = i_;

  // Generate single clause
  auto get_clause_ = [&rng_, &indexes_, k]() {
    // Initialize clause
    clause_type clause_;

    // Shuffle list of indexes
    std::shuffle(std::begin(indexes_), std::end(indexes_), rng_);

    // Update clause
    for( size_t j = 0; j < k; ++j){
        clause_.add_lit(pysa::sat::Lit_WS(indexes_[j], rng_()%2));
    }

    // Return clauses
    return clause_;
  };

  // Initialize clauses
  pysa::sat::Formula_WS formula;
  for( size_t i = 0; i < m; ++i){
    formula.add_clause(get_clause_());
  }
  return formula;
}

int main(){
    std::size_t max_steps = 100000;
    unsigned long max_unsat = 0;
    float p = 0.5;
    std::uint64_t seed = 1234;

    // Get random SAT problem
    int k = 3;
    int n = 21;
    int m = 60;
    auto formula = GetRandomInstance(k, n, m);
    pysa::sat::WalkSatOptimizer wsopt(formula, seed, p);
    wsopt.restart_state();
    size_t nsteps = 0;
    size_t n_unsat = formula.num_clauses();
    for(; nsteps < max_steps; ++nsteps){
        n_unsat = wsopt.step();
        if(n_unsat<=max_unsat)
            break;
    }
    std::cout << "Converged after "<< nsteps << " steps.\n";
    assert(n_unsat==0);


}
