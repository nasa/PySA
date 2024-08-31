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
#include <set>

#include "algstd/sat/dtypes.hpp"
#include "algstd/sat/clause.hpp"
#include "algstd/sat/formula.hpp"

namespace pysa::algstd {

  template<typename ClauseT=SATClauseV<uint32_t>>
  struct SATProp {
    //! Wrapper class to perform unit propagation on a SAT formula.
    typedef SATFormula<ClauseT> FormulaT;
    typedef ClauseT clause_type;
    typedef typename FormulaT::literal_type literal_type;
    typedef typename FormulaT::clause_container clause_container;
    typedef literal_type conflict_node_type;
    typedef std::pair<conflict_node_type, conflict_node_type> conflict_edge_type;
    typedef std::tuple<literal_type, size_t, literal_type> implication_type;
    typedef std::vector<implication_type> implication_seq;

    SATFormula<ClauseT> _formula;
    SATIndexer _cv;
    const size_t n_vars;
    // Literal value used to represent a conflict
    const literal_type conflict_lit;
    // Queue of units
    std::vector<literal_type> units;
    std::vector<uint8_t> _state;
    // Decided variables
    std::vector<uint8_t> _decided_vars;
    // First decision levels of clauses
    std::vector<int64_t> _clause_decs;
    // First satisfying literal for each clause
    std::vector<literal_type> _first_sat_lit;
    // First literal considered for each clause
    std::vector<literal_type> _first_vis_lit;
    size_t _unit_pos;
    size_t _unit_n;
    // Decision level of a variable
    std::vector<int64_t> _var_dec_lev;
    int64_t _conflict_backtrack_lev;

    // The antecedent unit clause of a consequent literal
    std::vector<ClIdx> _antecedent;
    // The consequent literal of a unit clause
    std::vector<literal_type> _consequent;

    // A copy of the consequent array when a conflict was reached.
    std::vector<literal_type> _consequent_conflict;
    // A copy of the variable decision levels when a conflict was reached.
    std::vector<int64_t> _var_dec_lev_conflict;
    //
    std::vector<literal_type> conflict_children;
    // Sequence of implications from previous call to unit_propagation()
    implication_seq imps;

    template<typename FormulaInit>
    explicit SATProp(FormulaInit &&formula):
        _formula(formula),
        _cv(_formula),
        n_vars(_cv.n_vars()),
        conflict_lit(n_vars, false),
        units(n_vars+1),
        _state(n_vars),
        _decided_vars(n_vars),
        _clause_decs(_cv.n_clauses(), INT64_MAX),
        _var_dec_lev(n_vars, -1),
        _var_dec_lev_conflict(n_vars),
        _first_sat_lit(_cv.n_clauses()),
        _first_vis_lit(_cv.n_clauses()),
        _antecedent(n_vars),
        _consequent(_cv.n_clauses()),
        _consequent_conflict(_cv.n_clauses()),
        _unit_pos{0},
        _unit_n{0}{

    }

    void initialize() {
      clause_container &clauses = _formula.clauses();
      //units.clear();
      imps.clear();
      size_t ncl = _formula.num_clauses();

      for (size_t i = 0; i < ncl; ++i) {
        clause_type &cl = clauses[i];
        if (is_unit(cl)) {
          size_t v = cl[0].idx();
          if(_antecedent[v].is_null()) {
            add_unit(cl[0]);
            _antecedent[v] = _cv.clauses_by_var[v].back();
            _consequent[i] = cl[0];
          }
        }
      }
    }

    void add_unit(literal_type u) {
      units[_unit_n++] = u;
      size_t v = u.idx();
      assert(_unit_n <= units.size());
      assert(_antecedent[v].is_null());
    }

    void add_clause(ClauseT &&cl) {
      _formula.add_clause(std::move(cl));
      _cv.include_clause(_formula._clauses.back());
      _clause_decs.push_back(INT64_MAX);
      _first_sat_lit.emplace_back();
      _first_vis_lit.emplace_back();
      if(cl.size() > 1){
        _consequent.emplace_back();
        _consequent_conflict.emplace_back();
      } else {
        size_t v = cl[0].idx();
        if(_antecedent[v].is_null()) {
          // add the literal of a learned unit clause if it does not yet have an antecedent
          add_unit(cl[0]);
          _clause_decs.back() = 0;
          _antecedent[v] = _cv.clauses_by_var[v].back();
          _consequent.push_back(cl[0]);
          _consequent_conflict.push_back(cl[0]);
        } else {
          assert(_antecedent[v].sign() == cl[0].sign());
          _consequent.emplace_back();
          _consequent_conflict.emplace_back();
        }
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
            assert(_consequent[i].is_null());
            _consequent[i] = conflict_lit;
            unsat = true;
#if defined(SATTRACE)
            std::cout << "\033[1;31m" << "!U" << "\033[0m\n";
#endif
          } else {
            if (is_unit(cl)) {
              literal_type implied_lit = cl[0];
              size_t impl_var = implied_lit.idx();
              // Add unit to stack if it does not already have an antecedent
              // Can occur if UP reduces 2 clauses to the same unit clause in this loop
              if(_antecedent[impl_var].is_null() && _consequent[i].is_null()) {
                add_unit(implied_lit);
                _antecedent[impl_var] = cli;
                _consequent[i] = implied_lit;
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
            std::cout << "\033[1;32m" << " S " << i  << "\033[0m\n";
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
      _conflict_backtrack_lev = INT64_MAX;
      std::copy(_consequent.begin(), _consequent.end(), _consequent_conflict.begin());
      std::copy(_var_dec_lev.begin(), _var_dec_lev.end(), _var_dec_lev_conflict.begin());

    }

    std::pair<int64_t, bool> unit_prop_reverse(int64_t conflict_level, int64_t decision_level = 0) {
      // The earliest decision level > 0 across all the propagated clauses
      int64_t earliest_dec = decision_level;

      bool conflict_cut = false;
      assert(_unit_pos > 0);
      clause_container &clauses = _formula.clauses();
      literal_type unit_lit = units[--_unit_pos];
      size_t var = unit_lit.idx();
      ClIdx unit_lit_antecedent = _antecedent[var];
      if(conflict_level == decision_level)
        conflict_children.clear();
      assert(_var_dec_lev[var] == decision_level);
      _var_dec_lev[var] = -1;
      if(!unit_lit_antecedent.is_null()){
        earliest_dec = _clause_decs[unit_lit_antecedent.idx()];
        assert(earliest_dec != INT64_MAX);
      }

      assert(_decided_vars[var]);
      _decided_vars[var] = 0;
      std::vector<ClIdx> &to_clauses = _cv.clauses_by_var[unit_lit.idx()];
      for (size_t _i = to_clauses.size(); _i > 0; _i--) {
        ClIdx cli = to_clauses[_i - 1];
        size_t i = cli.idx();
        clause_type &cl = clauses[i];
        if (cli.sign() != unit_lit.sign() && _first_sat_lit[i].is_null()) {
          // ~ l in an unsatisfied clause

          if (is_empty(cl)) { // clause was refuted by this unit
            assert(_consequent[i] == conflict_lit);
            _consequent[i] = literal_type();
            _conflict_backtrack_lev = std::min(_conflict_backtrack_lev, _clause_decs[i]);
#if defined(SATTRACE)
            std::cout << "XU\n";
#endif
          } else {
            if (is_unit(cl)) { //
              literal_type implied_lit = cl[0];
              size_t impl_var = implied_lit.idx();
              if(_antecedent[impl_var] == cli && _consequent[i] == implied_lit){
                remove_unit();
                _antecedent[impl_var] = ClIdx();
                assert(units[_unit_n] == cl[0]);
#if defined(SATTRACE)
                std::cout << "XA " << i << " . " << implied_lit << "\n";
#endif
              }
              _consequent[i] = literal_type();
            }
          }
          // check if this clause later becomes an antecedent
          if (!_consequent_conflict[i].is_null()) {
            literal_type _consequent_lit = _consequent_conflict[i];
            if(conflict_level == decision_level) {
#if defined(SATTRACE)
              std::cout << "CG " << unit_lit << " -> " << _consequent_lit << '\n';
#endif
              conflict_children.push_back(_consequent_lit);
            }
            if (conflict_level > decision_level
                && (_consequent_lit == conflict_lit
                    || _var_dec_lev_conflict[_consequent_lit.idx()] == conflict_level)) {
              conflict_cut = true;
            }
          }
          if (_clause_decs[i] == decision_level) {
            _clause_decs[i] = INT64_MAX;
          }
          cl.add_lit(~unit_lit);
        } else { // l in the clause
          if (_first_sat_lit[i] == unit_lit) {// This literal was the first to satisfy the clause
            _first_sat_lit[i] = literal_type();
#if defined(SATTRACE)
            std::cout << "XS " << i << "\n";
#endif
          }
        }
      }
#if defined(SATTRACE)
      std::cout << "XL " << unit_lit << "\n";
#endif

      return {earliest_dec, conflict_cut};
    }
  };

}