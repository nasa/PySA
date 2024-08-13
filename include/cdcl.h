#ifndef PYSA_CDCL_CDCL_H
#define PYSA_CDCL_CDCL_H

#include <cassert>
#include <iostream>
#include <utility>
#include "algstd/sat.hpp"

typedef pysa::algstd::SATClauseV<uint32_t> ClauseT;
typedef ClauseT::literal_type Lit;
typedef pysa::algstd::SATFormula<ClauseT> FormulaT;

static const int CDCLSAT = 0;
static const int CDCLUNSAT = -1;
static const int CDCLCONFLICT = 1;

struct CDCLConflict {
  int64_t conflict_level = INT64_MAX;
  int64_t decision_level = INT64_MAX;
  std::vector<Lit> decision_cut;
  ClauseT learned_clause;
};

struct CDCL {
  CDCL(FormulaT &&formula)
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

  int run();

  FormulaT formula;
  pysa::algstd::SATProp<ClauseT> prop;
  // Current state vector
  std::vector<uint8_t> _state;
  CDCLConflict current_conflict;
  std::vector<CDCLConflict> conflicts;

  // Current number of unsat clauses
  int64_t _level;
};
#endif //PYSA_CDCL_CDCL_H
