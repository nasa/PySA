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

#pragma once

#include <cassert>
#include <iostream>
#include <utility>
#include "algstd/satprop.hpp"

typedef pysa::algstd::SATClauseV<uint32_t> ClauseT;
typedef ClauseT::literal_type Lit;
typedef pysa::algstd::SATFormula<ClauseT> FormulaT;

static const int CDCLSAT = 0;
static const int CDCLUNSAT = -1;
static const int CDCLCONFLICT = 1;

struct CDCLConflict {
  /// Decision level of the encountered conflict
  int64_t conflict_level = INT64_MAX;
  /// Decision level to backtrack to
  int64_t decision_level = INT64_MAX;
  /// Decided literal leading to the conflict
  Lit decision_lit;
  /// Unique implication point (UIP) literal at the conflict decision level.
  /// That is, all implication paths from decision_lit to the conflict contain uip_lit.
  Lit uip_lit;
  /// The reason cut of literals leading to the conflict.
  std::vector<Lit> decision_cut;
  /// The final learned clause, obtained from the negation of the reason cut literals.
  ClauseT learned_clause;
};


struct CDCL {
  explicit CDCL(FormulaT &&formula)
      : formula(formula),
        prop(this->formula),
        conflicts(),
        _state(prop._cv.n_vars()),
        _level{0}{
    init();
  }

  void init() {
    prop.initialize();
  }

  void unsat_backtrack(size_t backtrack_steps) {
    for (size_t i = 0; i < backtrack_steps; ++i) {
      prop.unit_prop_reverse(-1, _level);
    }
  }

  std::optional<CDCLConflict> handle_conflict(size_t backtrack_steps, bool first);

  CDCLConflict end_conflict() {
    CDCLConflict c(std::move(current_conflict));
    current_conflict = CDCLConflict();
    return c;
  }

  int _run();
  int run();

  FormulaT formula;
  pysa::algstd::SATProp<ClauseT> prop;
  bool _uip = true;
  // Current state vector
  std::vector<uint8_t> _state;
  CDCLConflict current_conflict;
  std::vector<pysa::algstd::SATProp<ClauseT>::conflict_edge_type> _conflict_graph_edges;
  std::vector<Lit> _conflict_graph_nodes;
  std::vector<int64_t> _conflict_graph_earliest_dls;
  std::vector<CDCLConflict> conflicts;

  // Current number of unsat clauses
  int64_t _level;
};


