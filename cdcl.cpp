
#include "cdcl.h"

int CDCL::run() {
  size_t _init_pos = prop._unit_pos;
  size_t _init_n = prop._unit_n;
  size_t _prop_steps = 0;
  bool _prop_unsat = false;
  while (prop._unit_pos < prop._unit_n) { // Unit propagation
    _prop_steps++;
    pysa::algstd::ClFlag clf = prop.unit_prop_step(_level);
    if (int(pysa::algstd::ClFlag::UNIT & clf)) {
      if (int(clf & pysa::algstd::ClFlag::UNSAT)) {
        // Contradicting units were propagated
        _prop_unsat = true;
        break;
      } else { // Should not happen
        throw std::runtime_error("Literal was propagated twice.");
      }
    }
    if (clf == pysa::algstd::ClFlag::UNSAT) { // Conflict found at this decision level
      auto _hc = handle_conflict(_prop_steps, true);
      if (current_conflict.decision_level == _level) { //
        return CDCLUNSAT;
      }
      assert(_init_pos == prop._unit_pos);
      assert(_init_n == prop._unit_n);
      if(_level > 0)
        return CDCLCONFLICT;
      else {
#if defined(SATTRACE)
        std::cout << "\033[1;31m" << "unsat up0" << "\033[0m\n";
#endif
        return CDCLUNSAT;
      }
    }
  }
  if (_prop_unsat) {
    unsat_backtrack(_prop_steps);
    return CDCLUNSAT;
  }
  if (prop.satisfied()) {
    return CDCLSAT;
  }
  // Decide on the next branching variable
  Lit next_lit = Lit();
  for(size_t i = 0; i < prop._cv.n_vars(); ++i){
    if(!prop._decided_vars[i]){
      next_lit = Lit(i, false);
      break;
    }
  }
  assert(!next_lit.is_null());

  //CDCLConflict conflicts[2];
  for (int b = 0; b < 2; ++b) { // Branch

    Lit _prop_lit;
    if (b > 0)
      _prop_lit = next_lit;
    else
      _prop_lit = ~next_lit;
    _level++;
    prop.add_unit(_prop_lit);
#if defined(SATTRACE)
    std::cout << " D " << _level << " . " << _prop_lit << "\n";
#endif
    int rc = run();

    if (rc == CDCLCONFLICT) { // Resolve a conflict
      prop.remove_unit();
      _level--;
      auto _hc = handle_conflict(_prop_steps, false);
      if (_hc) {
        assert(_init_pos + _prop_steps == prop._unit_pos);
        assert(_init_n == prop._unit_n);
        // A conflict was handled down the stack.
        // ** End Backtrack **
        // Try this decision level again with learned clause.
        conflicts.push_back(*_hc);
        if(_hc->learned_clause.size() == 1 && _hc->learned_clause[0] == ~_prop_lit){
          // Identify if we learned a singleton clause refuting this decision
#if defined(SATTRACE)
          std::cout << "XDH " << _level << " . " << _prop_lit << "\n";
#endif
          continue;
        } else {
          b--;
          continue;
        }
      } else {
        assert(_init_pos == prop._unit_pos);
        assert(_init_n == prop._unit_n);
        // continue backtracking
        if (_level > 0) {
#if defined(SATTRACE)
          std::cout << "BD " << _level << " . " << _prop_lit << "\n";
#endif
          return CDCLCONFLICT;
        } else { // cannot backtrack further
          throw std::runtime_error("Backtracked beyond first decision level.");
        }
      }
    } else if (rc == CDCLUNSAT) {
      // No satisfying assignment in this branch.
#if defined(SATTRACE)
      std::cout << "XDU " << _level << " . " << _prop_lit << "\n";
#endif
      prop.remove_unit();
      _level--;
      continue;
    } else { //CDCLSAT
      _level--;
      return CDCLSAT;
    }
  }
  // At this point in the control flow, neither branch has a satisfying assignment
  unsat_backtrack(_prop_steps);

#if defined(SATTRACE)
  std::cout << "\033[1;31m" << "unsat D " << _level+1 << "\033[0m\n";
#endif

  return CDCLUNSAT;
}

std::optional<CDCLConflict> CDCL::handle_conflict(size_t backtrack_steps, bool first) {
  if (first) {// initialize conflict
    current_conflict = CDCLConflict{};
    current_conflict.conflict_level = _level;
    prop.begin_conflict();
  }
  size_t num_preds = 0;
  ClauseT learned_clause;
  if (_level == current_conflict.decision_level - 1) {
    // We have finished handling the earliest decision level preceding this conflict.
    // Learn a new clause to prevent this conflict
    ClauseT cl;
    for (Lit l: current_conflict.decision_cut)
      cl.add_lit(~l);
#if defined(SATTRACE)
    std::cout << "\033[1;33m" << "!C " << prop._cv.n_clauses() << " . " << cl << " 0\033[0m\n";
#endif
    current_conflict.learned_clause = cl;
    prop.add_clause(std::move(cl));
    return {end_conflict()};
  }
  for (size_t i = 0; i < backtrack_steps; ++i) {
    Lit unit_lit = prop.units[prop._unit_pos - 1];
    auto u = prop.unit_prop_reverse(current_conflict.conflict_level, _level);
    if (first)
      // at the conflict decision level, identify the earliest decision level
      // with a preceeding vertex.
      current_conflict.decision_level = std::min(current_conflict.decision_level, u.first);
    num_preds += u.second;
    if (first && (i == backtrack_steps - 1)) {
      // This was the decided variable at the conflict's decision level
      current_conflict.decision_cut.push_back(unit_lit);
    }
    if (!first && u.second) {
      // This literal is the parent to a consequent in the conflict level
      current_conflict.decision_cut.push_back(unit_lit);
    }
  }

  return {};
}
