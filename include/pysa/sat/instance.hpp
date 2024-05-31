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

#include <algorithm>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

#include "../bitset/bitset.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace pysa::dpll::sat {

auto GetRandomInstance(const std::size_t k, const std::size_t n,
                       const std::size_t m,
                       const std::optional<std::size_t> seed = std::nullopt) {
  /*
   * Generate k-SAT random instance.
   */
  using clause_type = std::vector<int>;

  // Check
  if (n < k) throw std::runtime_error("'n' must be larger than 'k'");

  // Get random seed
  const std::size_t seed_ = seed.value_or(std::random_device()());

  // Initialize random generator
  std::mt19937_64 rng_(seed_);

  // Initialize set of indexes
  std::vector<int> indexes_(n);
  for (std::size_t i_ = 0; i_ < n; ++i_) indexes_[i_] = i_ + 1;

  // Generate single clause
  auto get_clause_ = [&rng_, &indexes_, k]() {
    // Initialize clause
    clause_type clause_;

    // Shuffle list of indexes
    std::shuffle(std::begin(indexes_), std::end(indexes_), rng_);

    // Update clause
    std::transform(std::begin(indexes_), std::begin(indexes_) + k,
                   std::back_inserter(clause_),
                   [&rng_](auto &&x) { return (rng_() % 2 ? 1 : -1) * x; });

    // Return clauses
    return clause_;
  };

  // Initialize clauses
  std::vector<clause_type> clauses_;
  std::generate_n(std::back_inserter(clauses_), m, get_clause_);
  return clauses_;
}

template <typename Formula>
auto GetNVars(Formula &&formula) {
  std::size_t max_ = 0;
  for (const auto &clause_ : formula)
    for (const auto &x_ : clause_) {
      // All variables should be different from zero
      if (x_ == 0)
        throw std::runtime_error("All variables should be different from zero");
      if (const std::size_t a_ = std::abs(x_); a_ > max_) max_ = a_;
    }
  return max_;
}

template <typename BitSet = BitSet<>,
          template <typename...> typename Vector = std::vector>
struct Instance {
  // VarIndex -> ClauseIndex
  const Vector<Vector<std::size_t>> clauses;

  // VarIndex -> VarSign in Clause
  const Vector<BitSet> signs;

  // ClauseIndex -> VarIndex
  const Vector<std::size_t> last_var;

  template <typename Clauses, typename Signs, typename LastVar>
  Instance(Clauses &&clauses, Signs &&signs, LastVar &&last_var)
      : clauses{std::forward<Clauses>(clauses)},
        signs{std::forward<Signs>(signs)},
        last_var{std::forward<LastVar>(last_var)} {}

  /**
   * @brief Construct the instance from a formula.
   * @tparam Formula An iterable type with elements of type Clause.
   *  A Clause is an iterable type with elements of type Literal.
   *  A Literal is a type that can be cast to a signed integer.
   * @param formula An object of type Formula.
   */
  template <typename Formula>
  explicit Instance(const Formula &formula) : Instance{_Build(formula)} {}

  std::size_t n_vars() const {
    /*
     * Return number of variables.
     */
    assert(std::size(clauses) == std::size(signs));
    return std::size(clauses);
  }

  std::size_t n_clauses() const {
    /*
     * Return number of clauses.
     */
    return std::size(last_var);
  }

  bool operator==(const Instance &other) const {
    /*
     * Check equality.
     */
    return clauses == other.clauses && signs == other.signs &&
           last_var == other.last_var;
  }

 private:
  template <typename Formula>
  static constexpr auto _Build(const Formula &formula) {
    // Check all variables are different from zero
    for (const auto &cl_ : formula)
      for (const auto &x_ : cl_)
        if (x_ == 0)
          throw std::runtime_error(
              "All variables should be different from zero");

    // Get number of variables
    const auto n_ = GetNVars(formula);

    // Get number of constraints
    const auto m_ = std::size(formula);

    // Initialize
    std::decay_t<decltype(clauses)> clauses_(n_);
    std::decay_t<decltype(signs)> signs_(n_);
    std::decay_t<decltype(last_var)> last_var_;

    // Update clauses and signs
    for (std::size_t i_ = 0; i_ < m_; ++i_) {
      // Get clause
      const auto &clause_ = formula[i_];

      // For each variable
      for (const auto &x_ : clause_) {
        // Get position of variable
        std::size_t pos_ = std::abs(x_) - 1;

        // Check
        if (pos_ >= n_)
          throw std::runtime_error(
              "Variable index is larger than the number of variables");

        // Update clauses
        clauses_[pos_].push_back(i_);

        // Update signs
        signs_[pos_].push_back(x_ < 0);
      }
    }

    // Update last variable
    for (const auto &clause_ : formula)
      last_var_.push_back(std::transform_reduce(
                              std::begin(clause_), std::end(clause_), 0,
                              [](auto &&x, auto &&y) { return std::max(x, y); },
                              [](auto &&x) { return std::abs(x); }) -
                          1);

    // Return new instance
    return Instance{clauses_, signs_, last_var_};
  }
};
}  // namespace pysa::dpll::sat
