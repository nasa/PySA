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

//! Data structures for satisfiability problems

#pragma once

#include <iostream>
#include <vector>
#include <regex>
#include <sstream>
#include <cassert>

#include "algstd/sat/dtypes.hpp"
#include "algstd/sat/clause.hpp"

namespace pysa::algstd {
  template<typename ClauseT>
  struct SATFormula {
    //! k-SAT formula, where the maximum clause size k is specified at compile time
    typedef ClauseT clause_type;
    typedef typename clause_type::literal_type literal_type;
    typedef typename std::vector<ClauseT> clause_container;

    std::vector<ClauseT> _clauses;

    //size_t _num_lits;
    SATFormula() = default;

    SATFormula(const SATFormula &) = default;

    SATFormula(SATFormula &&) = default;

    SATFormula &operator=(const SATFormula &) = default;

    SATFormula &operator=(SATFormula &&) = default;

    SATFormula(std::initializer_list<clause_type> li){
      for(const clause_type& cl : li){
        add_clause(cl);
      }
    }
    clause_container &clauses() {
      return _clauses;
    }

    const clause_container &clauses() const {
      return _clauses;
    }

    size_t num_clauses() const {
      return _clauses.size();
    }

    void add_clause(clause_type &&clause) {
      _clauses.emplace_back(clause);
    }
    void add_clause(const clause_type &clause) {
      _clauses.emplace_back(clause);
    }

    clause_type &operator[](size_t i) {
      return _clauses[i];
    }

    const clause_type &operator[](size_t i) const {
      return _clauses[i];
    }

    clause_type remove_clause(size_t idx) {
      if (idx != _clauses.size() - 1)
        std::swap(_clauses[idx], _clauses.back());
      clause_type cl = std::move(_clauses.back());
      _clauses.pop_back();
      return cl;
    }
  };


  struct SATIndexer {
    //! Map a variable index to the list of clauses it participates in, along with negations.
    std::vector<std::vector<ClIdx>> clauses_by_var;
    size_t _ncl = 0;

    template<typename ClauseT>
    void include_clause(const ClauseT &cl) {
      typedef typename ClauseT::literal_type LitT;
      for (const LitT &lit: cl) {
        size_t v = lit.idx();
        bool s = lit.sign();
        if (v >= clauses_by_var.size())
          clauses_by_var.resize(v + 1);
        clauses_by_var[v].emplace_back(_ncl, s);
      }
      _ncl++;
    }

    size_t n_vars() const {
      return clauses_by_var.size();
    }

    size_t n_clauses() const {
      return _ncl;
    }

    template<typename FormulaT>
    SATIndexer(const FormulaT &formula): clauses_by_var{} {
      typedef typename FormulaT::clause_type ClauseT;
      const auto &clauses = formula.clauses();

      for (const ClauseT &cl: clauses) {
        include_clause(cl);
      }
    }
  };

  template<typename ClauseT=SATClauseV<uint32_t>>
  struct SATProp {
    //! Wrapper class to perform unit propagation on a SAT formula.
    typedef SATFormula<ClauseT> FormulaT;
    typedef ClauseT clause_type;
    typedef typename FormulaT::literal_type literal_type;
    typedef typename FormulaT::clause_container clause_container;
    typedef std::tuple<literal_type, size_t, literal_type> implication_type;
    typedef std::vector<implication_type> implication_seq;

    SATFormula<ClauseT> _formula;
    SATIndexer _cv;
    // Queue of units
    std::vector<literal_type> units;
    std::vector<uint8_t> _state;
    // Decided variables
    std::vector<uint8_t> _decided_vars;
    // First decision levels of clauses
    std::vector<size_t> _clause_decs;
//      std::vector<size_t> _pred_stack;
//      std::vector<size_t> _pred_cl;
    size_t _pred_n;
    // First satisfying literal for each clause
    std::vector<literal_type> _first_sat_lit;
    // First literal considered for each clause
    std::vector<literal_type> _first_vis_lit;
    size_t _unit_pos;
    size_t _unit_n;
    // Decision level of a variable
    std::vector<int64_t> _var_dec_lev;
    std::vector<ClIdx> _antecedent;
    // The consequent literal of a unit clause
    std::vector<literal_type> _consequent;
    // A copy of the consequent array when a conflict was reached.
    std::vector<literal_type> _consequent_conflict;
    // A copy of the variable decision levels when a conflict was reached.
    std::vector<int> _var_dec_lev_conflict;
    // Sequence of implications from previous call to unit_propagation()
    implication_seq imps;

    // Literal value used to represent a conflict
    const literal_type conflict_lit;

    template<typename FormulaInit>
    SATProp(FormulaInit &&formula):
        _formula(formula),
        _cv(_formula),
        units(_cv.n_vars()+1),
        _state(_cv.n_vars()),
        _decided_vars(_cv.n_vars()),
        _clause_decs(_cv.n_clauses(), SIZE_T_MAX),
//            _pred_stack(_cv.n_vars()),
//            _pred_cl(_cv.n_clauses()),
        _var_dec_lev(_cv.n_vars(), -1),
        _var_dec_lev_conflict(_cv.n_vars()),
        _first_sat_lit(_cv.n_clauses()),
        _first_vis_lit(_cv.n_clauses()),
        _antecedent(_cv.n_vars()),
        _consequent(_cv.n_clauses()),
        _consequent_conflict(_cv.n_clauses()),
        _unit_pos{0},
        _unit_n{0},
        conflict_lit(_cv.n_vars(), false) {

    }

    void initialize() {
      clause_container &clauses = _formula.clauses();
      //units.clear();
      imps.clear();
      size_t ncl = _formula.num_clauses();

      for (size_t i = 0; i < ncl; ++i) {
        clause_type &cl = clauses[i];
        if (is_unit(cl)) {
          add_unit(cl[0]);
          _consequent[i] = cl[0];
        }
      }
    }

    void add_unit(literal_type u) {
      units[_unit_n++] = u;
      assert(_unit_n <= units.size());
    }

    void add_clause(ClauseT &&cl) {
      _formula.add_clause(std::move(cl));
      _cv.include_clause(_formula._clauses.back());
      _clause_decs.push_back(SIZE_T_MAX);
      // _pred_cl.push_back(0);
      _first_sat_lit.emplace_back();
      _first_vis_lit.emplace_back();
      if(cl.size() > 1){
        _consequent.emplace_back();
        _consequent_conflict.emplace_back();
      } else {
        _consequent.push_back(cl[0]);
        _consequent_conflict.push_back(cl[0]);
      }
    }

    void remove_unit() {
      --_unit_n;
    }

    bool satisfied() {
      for (size_t cli = 0; cli < _cv.n_clauses(); ++cli) {
        if (_first_sat_lit[cli].is_null())
          return false;
      }
      return true;
    }

    ClFlag unit_prop_step(int64_t decision_level = 0) {
      assert(_unit_pos < _unit_n);

      bool unsat = false;
      clause_container &clauses = _formula.clauses();
      literal_type unit_lit = units[_unit_pos++];
      size_t var = unit_lit.idx();
      const auto &to_clauses = _cv.clauses_by_var[unit_lit.idx()];
      if (_var_dec_lev[var] != -1) { // This variable was already decided
        if (unit_lit.sign() ^ (_state[var] > 0)) {
          return ClFlag::UNSAT | ClFlag::UNIT;
        } else {
          return ClFlag::UNIT;
        }
      }
      _var_dec_lev[var] = decision_level;
      _decided_vars[var] = 1;
      if(unit_lit.sign()){
        _state[var] = 0;
      } else {
        _state[var] = 1;
      }
#if defined(SATTRACE)
      std::cout << " L " << unit_lit << "\n";
#endif
      for (ClIdx cli: to_clauses) {
        size_t i = cli.idx();
        clause_type &cl = clauses[i];
        if (cli.sign() != unit_lit.sign() && _first_sat_lit[i].is_null()) {
          // ~ l in an unsatisfied clause
          auto prop_flag = propagate_lit(cl, unit_lit);
#if defined(SATTRACE)
          std::cout << " c " << i << " . " << cl << "\n";
#endif
          assert(prop_flag == ClFlag::CLSIMP);
          if (decision_level < _clause_decs[i]) {
            _clause_decs[i] = decision_level;
          }
          if (_first_vis_lit[i].is_null())
            _first_vis_lit[i] = unit_lit;
          if (is_empty(cl)) { // clause was refuted by this unit
            //imps.emplace_back(unit_lit, i, literal_type());
            // size_t var = unit_lit.idx();
            //_consequent[i] = conflict_lit;
            assert(_consequent[i] == ~unit_lit);
            unsat = true;
#if defined(SATTRACE)
            std::cout << "\033[1;31m" << "!U" << "\033[0m\n";
#endif
          } else {
            if (is_unit(cl)) {
              literal_type implied_lit = cl[0];
              _consequent[i] = implied_lit;
              size_t impl_var = implied_lit.idx();
              // Add unit to stack if it does not already have an antecedent
              // Can occur if UP reduces 2 clauses to the same unit clause in this loop
              if(_antecedent[impl_var].is_null()) {
                add_unit(implied_lit);
                _antecedent[impl_var] = cli;
#if defined(SATTRACE)
                std::cout << "\033[1;34m"
                  << " A " << i << " . " << implied_lit
                  << "\033[0m\n";
#endif
              }
            }
          }

        } else {
          if (cli.sign() == unit_lit.sign() && _first_sat_lit[i].is_null()) {
            // l is first to satisfy this clause not yet satisfied
            _first_sat_lit[i] = unit_lit;
#if defined(SATTRACE)
            //std::cout << "\033[1;32m" << " S " << i  << "\033[0m\n";
#endif
          }
        }
      }
      if (unsat)
        return ClFlag::UNSAT;
      else
        return ClFlag::NONE;
    }

    void begin_conflict() {
      std::copy(_consequent.begin(), _consequent.end(), _consequent_conflict.begin());
      std::copy(_var_dec_lev.begin(), _var_dec_lev.end(), _var_dec_lev_conflict.begin());
    }

    std::pair<int64_t, bool> unit_prop_reverse(int64_t conflict_level, int64_t decision_level = 0) {
      // The earliest decision level across all the propagated clauses
      int64_t earliest_dec = decision_level;
      // Number of conflict predecessors in this propagation, including the conflict clause
      size_t num_confl_preds = 0;
      bool conflict_cut = false;
      assert(_unit_pos > 0);
      clause_container &clauses = _formula.clauses();
      literal_type unit_lit = units[--_unit_pos];
      size_t var = unit_lit.idx();
      //assert(_var_dec_lev[var] == decision_level);
      if(_var_dec_lev[var] != decision_level){
        return {-1, false};
      }
      _var_dec_lev[var] = -1;

      assert(_decided_vars[var]);
      _decided_vars[var] = 0;
      std::vector<ClIdx> &to_clauses = _cv.clauses_by_var[unit_lit.idx()];
      for (size_t _i = to_clauses.size(); _i > 0; _i--) {
        ClIdx cli = to_clauses[_i - 1];
        size_t i = cli.idx();
        clause_type &cl = clauses[i];
        if (cli.sign() != unit_lit.sign() && _first_sat_lit[i].is_null()) {
          // ~ l in an unsatisfied clause
          if (_clause_decs[i] < earliest_dec) {
            earliest_dec = _clause_decs[i];
          }
          if (is_empty(cl)) { // clause was refuted by this unit
            assert(_consequent[i] == ~unit_lit);
#if defined(SATTRACE)
            std::cout << "XU\n";
#endif
          } else {
            if (is_unit(cl)) { // This clause is the antecedent for this unit
              literal_type implied_lit = cl[0];
              size_t impl_var = implied_lit.idx();
              if(_antecedent[impl_var] == cli){
                remove_unit();
                _antecedent[impl_var] = ClIdx();
                assert(units[_unit_n] == cl[0]);
#if defined(SATTRACE)
                std::cout << "XA " << i << " . " << implied_lit << "\n";
#endif
              }
              assert(_consequent[i] == cl[0]);
              _consequent[i] = literal_type();

            } else { // check if this clause later becomes an antecedent
              if (!_consequent_conflict[i].is_null()) {
                literal_type _consequent_lit = _consequent_conflict[i];
                if (conflict_level > decision_level
                    && _var_dec_lev_conflict[_consequent_lit.idx()] == conflict_level) {
                  conflict_cut = true;
                }
              }
            }
          }
          if (_clause_decs[i] == decision_level) {
            _clause_decs[i] = SIZE_T_MAX;
          }
          cl.add_lit(~unit_lit);
        } else { // l in the clause
          if (_first_sat_lit[i] == unit_lit) {// This literal was the first to satisfy the clause
            _first_sat_lit[i] = literal_type();
#if defined(SATTRACE)
            //std::cout << "XS " << i << "\n";
#endif
          }
        }
      }
#if defined(SATTRACE)
      std::cout << "XL " << unit_lit << "\n";
#endif
      return {earliest_dec, conflict_cut};
    }

    implication_seq &unit_propagation() {
      while (!units.empty()) {
        ClFlag clf = unit_prop_step();
        if (clf == ClFlag::UNSAT)
          break;
      }
      return imps;
    }
  };

  template<typename FormulaT>
  std::vector<std::tuple<typename FormulaT::literal_type, size_t>>
  unit_propagation(
      FormulaT &formula
  ) {
    //! Perform unit propagation on a formula and return the sequence of implications
    //! as (literal assignment, clause index) tuples.
    SATIndexer cv{formula};
    // Stack of units
    std::vector<size_t> _units;
    size_t ncl = formula.num_clauses();
    typename FormulaT::clause_container &clauses = formula.clauses();
    for (size_t i = 0; i < ncl; ++i) {
      typename FormulaT::clause_type &cl = clauses[i];
      if (cl.is_unit())
        _units.push_back(i);
    }
  }
}

template<typename T>
inline std::ostream &operator<<(std::ostream &os,
                                const std::vector<pysa::algstd::lidx<T>> &lits) {
  for (uint32_t i = 0; i < lits.size(); i++) {
    os << lits[i];
    if (i != lits.size() - 1)
      os << " ";
  }
  return os;
}

template<typename ClauseT>
inline std::ostream &operator<<(std::ostream &os,
                                const pysa::algstd::SATFormula<ClauseT> &f) {
  for(uint32_t i = 0; i < f.num_clauses(); ++i){
    os << f[i];
    os << " 0\n";
  }
  return os;
}

// Read in a literal from an istream
template<typename T>
inline std::istream &operator>>(std::istream &is, pysa::algstd::lidx<T> &lit) {
  int64_t l;
  if (is >> l) {
    bool s = l < 0;
    T v = (s ? -l : l) - 1;
    lit = pysa::algstd::lidx<T>(v, s);
  }
  return is;
}

namespace pysa::algstd {

  template<typename InputStream, typename ClauseType = SATClauseV<uint32_t>>
  SATFormula<ClauseType> ReadCNF(InputStream &&input_stream) {
    typedef typename ClauseType::literal_type literal_type;

    // Initialize formula
    SATFormula<ClauseType> formula;

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
      if (line.length() == 0)
        continue;

      // If line is a comment, skip
      if (std::tolower(line[0]) == 'c')
        continue;

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
        literal_type js;

        // For each token in line ...
        while (ss >> js) {
          // If token is different from zero, append to current clause
          if (!js.is_null())
            current_cl.add_lit(js);
            // Otherwise, append to formula
          else {
            formula.add_clause(std::move(current_cl));

            // Remove all elements from current_cl
            // (in case moving didn't reset it)
            current_cl = ClauseType();

            // Ignore everything after zero
            break;
          }
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
}