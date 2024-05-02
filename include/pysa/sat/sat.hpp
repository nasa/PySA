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
#include <list>
#include <memory>
#include <pysa/archive/archive.hpp>
#include <random>
#include <regex>
#include <sstream>
#include <vector>

#include "../bitset/bitset.hpp"
#include "../dpll/dpll.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace pysa::dpll::sat {

auto GetRandomInstance(const std::size_t k, const std::size_t n,
                       const std::size_t m, const std::size_t seed = 0) {
  /*
   * Generate k-SAT random instance.
   */
  using clause_type = std::vector<int>;

  // Check
  if (n < k) throw std::runtime_error("'n' must be larger than 'k'");

  // Get random seed
  const std::size_t seed_ = seed ? seed : std::random_device()();

  // Initialize random generator
  std::mt19937_64 rng_(seed_);

#ifndef NDEBUG
  std::cerr << "# Used seed: " << seed_ << std::endl;
#endif

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

template <typename InputStream, typename ClauseType = std::vector<int32_t>,
          template <typename...> typename Vector = std::vector>
auto ReadCNF(InputStream &&input_stream) {
  // Initialize formula
  Vector<ClauseType> formula;

  // Initialize current clause
  ClauseType current_cl;

  bool parsed_problem_line = false;
  std::size_t line_num = 0;
  std::string line;

  // For each line in input ...
  while (std::getline(input_stream, line)) {
    // Increment line number
    ++line_num;

    // If line is empty, skip
    if (line.length() == 0) continue;

    // If line is a comment, skip
    if (std::tolower(line[0]) == 'c') continue;

    // If line is problem line, skip
    if (std::regex_match(line, std::regex(R"(^p\s+cnf\s+\d+\s+\d+\s*)"))) {
      if (parsed_problem_line) {
        std::cerr << "Problem line parsed multiple times" << std::endl;
        throw std::runtime_error("CNF Parse Error");
      }
      parsed_problem_line = true;

#ifndef NDEBUG
      std::cerr << line << std::endl;
#endif
      continue;
    }

    // Check if problem line appears exactly once
    if (!parsed_problem_line) {
      std::cerr << "CNF file in the wrong format" << std::endl;
      throw std::runtime_error("CNF Parse Error");
    }

    // Check that line is a sequence of numbers
    if (!std::regex_match(line,
                          std::regex(R"(^\s*-?\d+(\s+-?\d+)*\s+0\s*$)"))) {
      std::cerr << "Failed to parse line " << line_num << ": " << line << ".";
      throw std::runtime_error("CNF Parse Error");
    }

    //  Parse lines
    {
      std::stringstream ss(line);
      typename ClauseType::value_type js;

      // For each token in line ...
      while (ss >> js)

        // If token is different from zero, append to current clause
        if (js != 0) current_cl.push_back(js);

        // Otherwise, append to formula
        else {
          formula.push_back(std::move(current_cl));

          // Remove all elements from current_cl
          // (in case moving didn't reset it)
          current_cl.clear();

          // Ignore everything after zero
          break;
        }

      // Only spaces and non-printable characters should remain
      {
        std::string left_;
        ss >> left_;
        if (!ss.eof() ||
            std::any_of(std::begin(left_), std::end(left_), [](auto &&c) {
              return !std::isspace(c) && std::isprint(c);
            })) {
          std::cerr << "Failed to parse line " << line_num << ": " << line
                    << ".";
          throw std::runtime_error("CNF Parse Error");
        }
      }
    }
  }

  // Return formula
  return formula;
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

template <typename Instance, typename T, typename = std::void_t<>>
struct is_pointer_to : std::false_type {};

template <typename Instance, typename T>
struct is_pointer_to<
    Instance, T,
    std::void_t<
        decltype(std::is_same_v<std::decay_t<decltype(*std::declval<T>())>,
                                Instance>)>> : std::true_type {};

template <typename Instance, typename T>
static constexpr auto is_pointer_to_v = is_pointer_to<Instance, T>::value;

template <typename BitSet = BitSet<>,
          template <typename...> typename List = std::list,
          template <typename...> typename Vector = std::vector,
          typename Instance = Instance<BitSet, Vector>>
struct Branch {
  const std::shared_ptr<const Instance> instance{nullptr};
  const std::size_t max_n_unsat{0};

  // Empty branch
  Branch() {}

  template <
      typename InstancePtr,
      std::enable_if_t<is_pointer_to_v<Instance, InstancePtr>, bool> = true>
  Branch(InstancePtr instance, std::size_t max_n_unsat = 0)
      : instance{instance},
        max_n_unsat{max_n_unsat},
        _state(instance->n_vars()),
        _partial_sat(instance->n_clauses()),
        _pos{0},
        _n_unsat{0},
        _valid{true} {}

  template <typename Clauses,
            std::enable_if_t<!is_pointer_to_v<Instance, Clauses>, bool> = true>
  Branch(const Clauses &clauses, std::size_t max_n_unsat = 0)
      : Branch(std::make_shared<Instance>(clauses), max_n_unsat) {}

  template <typename InstancePtr>
  Branch(InstancePtr instance, std::size_t max_n_unsat, BitSet state,
         BitSet partial_sat, std::size_t pos, std::size_t n_unsat, bool valid)
      : instance{instance},
        max_n_unsat{max_n_unsat},
        _state{state},
        _partial_sat{partial_sat},
        _pos{pos},
        _n_unsat{n_unsat},
        _valid{valid} {}

  const BitSet &state() const { return _state; }

  std::size_t pos() const { return _pos; }

  std::size_t n_unsat() const { return _n_unsat; }

  std::size_t n_sat() const { return _partial_sat.count(); }

  const BitSet &partial_sat() const { return _partial_sat; }

  bool leaf() const {
    /*
     * Check if this node is a leaf.
     */
    // Check that number of sat/unsat match
    assert(_pos < std::size(_state) || !_valid ||
           instance->n_clauses() - n_sat() == n_unsat());

    // Check if leaf
    return _pos == std::size(_state);
  }

  bool partial() const {
    /*
     * Check if this node is a partial branch.
     */
    return !leaf() || !_valid;
  }

  void next() {
    // Get references
    auto &clauses_ = instance->clauses[_pos];
    auto &signs_ = instance->signs[_pos];

    // For each clause
    for (std::size_t i_ = 0, end_ = std::size(clauses_); i_ < end_; ++i_) {
      const auto cl_ = clauses_[i_];

      // Update the partial satisfiability by checking if this variable has the
      // right sign in clause
      _partial_sat.apply_or(signs_.test(i_) ^ _state.test(_pos), cl_);

      // If this variable is the last variable in clause, also update the
      // number of unsat
      _n_unsat += (instance->last_var[cl_] == _pos) && !_partial_sat.test(cl_);
    }

    // Update position
    if (_n_unsat <= max_n_unsat)
      ++_pos;
    else {
      _valid = false;
      _pos = std::size(_state);
    }
  }

  List<Branch> branch() const {
    // Only branch if this variable is set to zero
    if (_state.test(_pos)) return {};

    // Create a new branch
    auto branch_ = *this;

    // Set this variable to one
    branch_._state.set(_pos);

    // Return
    return {std::move(branch_)};
  }

  template <typename Buffer>
  auto dump() const {
    /*
     * Dump Branch.
     */
    auto buffer_ = pysa::branching::dump<Buffer>(max_n_unsat);
    buffer_ += pysa::branching::dump<Buffer>(_state);
    buffer_ += pysa::branching::dump<Buffer>(_partial_sat);
    buffer_ += pysa::branching::dump<Buffer>(_pos);
    buffer_ += pysa::branching::dump<Buffer>(_n_unsat);
    buffer_ += pysa::branching::dump<Buffer>(_valid);
    return buffer_;
  }

  bool operator==(const Branch &other) const {
    /*
     * Check equality.
     */
    return *instance == *other.instance && max_n_unsat == other.max_n_unsat &&
           _state == other._state && _partial_sat == other._partial_sat &&
           _pos == other._pos && _n_unsat == other._n_unsat &&
           _valid == other._valid;
  }

 private:
  BitSet _state;
  BitSet _partial_sat;
  std::size_t _pos;
  std::size_t _n_unsat;
  bool _valid;
};

template <typename StateType, typename SizeType>
struct Configuration {
  StateType state;
  SizeType n_unsat;

  Configuration() {}
  Configuration(const StateType &state, SizeType n_unsat)
      : state{state}, n_unsat{n_unsat} {}
  Configuration(StateType &&state, SizeType n_unsat)
      : state{std::move(state)}, n_unsat{n_unsat} {}

  auto hash() const {
    const auto seed_ = state.hash();
    return n_unsat + 0x9e3779b9 + (seed_ << 6) + (seed_ >> 2);
  }

  bool operator==(const Configuration &other) const {
    return state == other.state && n_unsat == other.n_unsat;
  }
};

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
              std::size_t n_threads = std::thread::hardware_concurrency(),
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
  return DPLL(init_, get_, verbose, n_threads, walltime, sleep_time);
}

#ifdef USE_MPI
namespace mpi {

template <typename MPI_Comm_World, typename Formula,
          typename SleepTime = decltype(60s),
          typename ThreadSleepTime = decltype(1ms)>
auto optimize(MPI_Comm_World &&mpi_comm_world, Formula formula,
              std::size_t max_n_unsat, bool verbose = false,
              std::size_t n_threads = std::thread::hardware_concurrency(),
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
      init_, get_, verbose, std::thread::hardware_concurrency(), {}, {}, {},
      std::tuple{instance_}, sleep_time, thread_sleep_time);
}

}  // namespace mpi
#endif

}  // namespace pysa::dpll::sat

namespace pysa::branching {

template <typename BitSet, template <typename...> typename List,
          template <typename...> typename Vector, typename Instance>
struct Archiver<pysa::dpll::sat::Branch<BitSet, List, Vector, Instance>> {
  using base_type = pysa::dpll::sat::Branch<BitSet, List, Vector, Instance>;

  template <typename Buffer>
  static constexpr auto dumps(const base_type &branch) {
    return branch.template dump<Buffer>();
  }

  template <typename Buffer, typename InstancePtr>
  static constexpr auto loads(typename Buffer::const_iterator buffer,
                              InstancePtr instance) {
    auto [h1_, max_n_unsat_] = load<std::size_t, Buffer>(buffer);
    auto [h2_, state_] = load<BitSet, Buffer>(h1_);
    auto [h3_, partial_sat_] = load<BitSet, Buffer>(h2_);
    auto [h4_, pos_] = load<std::size_t, Buffer>(h3_);
    auto [h5_, n_unsat_] = load<std::size_t, Buffer>(h4_);
    auto [h6_, valid_] = load<bool, Buffer>(h5_);
    return std::pair{
        h6_, base_type(instance, max_n_unsat_, std::move(state_),
                       std::move(partial_sat_), pos_, n_unsat_, valid_)};
  }
};

template <typename StateType, typename SizeType>
struct Archiver<pysa::dpll::sat::Configuration<StateType, SizeType>> {
  using base_type = pysa::dpll::sat::Configuration<StateType, SizeType>;

  template <typename Buffer>
  static constexpr auto dumps(const base_type &conf) {
    return dump<Buffer>(conf.state) + dump<Buffer>(conf.n_unsat);
  }

  template <typename Buffer>
  static constexpr auto loads(typename Buffer::const_iterator buffer) {
    using state_type = decltype(std::declval<base_type>().state);
    using n_unsat_type = decltype(std::declval<base_type>().n_unsat);
    auto [h1_, s_] = load<state_type, Buffer>(buffer);
    auto [h2_, n_] = load<n_unsat_type, Buffer>(h1_);
    return std::pair{h2_, base_type{s_, n_}};
  }
};

template <typename BitSet, template <typename...> typename List,
          template <typename...> typename Vector, typename Instance>
struct Archiver<
    std::list<pysa::dpll::sat::Branch<BitSet, List, Vector, Instance>>> {
  using base_type = pysa::dpll::sat::Branch<BitSet, List, Vector, Instance>;

  template <typename Buffer>
  static constexpr auto dumps(const std::list<base_type> &branches) {
    // Dump size
    auto buffer_ = dump<Buffer, std::size_t>(std::size(branches));

    // Dump branches
    for (const auto &br_ : branches) buffer_ += dump<Buffer>(br_);

    // Return buffer
    return buffer_;
  }

  template <typename Buffer, typename InstancePtr>
  static constexpr auto loads(typename Buffer::const_iterator buffer,
                              InstancePtr instance) {
    // Get size
    auto [buffer_, size_] = load<std::size_t, Buffer>(buffer);

    // Initialize branches
    std::list<base_type> branches_{};

    // Append all branches
    for (std::size_t i_ = 0; i_ < size_; ++i_) {
      // Get branch
      auto [h_, branch_] = load<base_type, Buffer>(buffer_, instance);

      // Append
      branches_.push_back(std::move(branch_));

      // Update head
      buffer_ = h_;
    }

    // Return branches
    return std::pair{buffer_, std::move(branches_)};
  }
};

}  // namespace pysa::branching
