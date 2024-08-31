#include <map>
#include "cdcl.h"

typedef pysa::algstd::SATProp<>::conflict_node_type conflict_node_type;
typedef typename pysa::algstd::SATProp<>::conflict_edge_type conflict_edge_type;


std::pair<Lit, std::vector<Lit>> analyze_uip_cut(
    const std::vector<conflict_edge_type>& edges,
    Lit conflict_lit,
    Lit decision_lit
    )
{
  /// Returns the precedent literals of the UIP cut in the DAG defined by a list of edges
  /// The UIP literal is the first element
  std::map<conflict_node_type, std::vector<conflict_node_type>> parents;
  for(const conflict_edge_type& e : edges){
    conflict_node_type s, t;
    std::tie(s,t) = e;
    parents[t].push_back(s);
  }
  std::vector<Lit> uip_cut;
  std::set<conflict_node_type> uip_set;
  Lit uip_lit;
  uip_set.insert(conflict_lit);
  assert(!edges.empty());

  for(const conflict_edge_type& e : edges){ // follow the edges to ensure backtrack order
    conflict_node_type s, t;
    std::tie(s,t) = e;
    if(auto _it = uip_set.find(t); _it != uip_set.end()){
      uip_set.erase(_it);
      if(t != conflict_lit)
        uip_cut.push_back(t);
      for( conflict_node_type p: parents[t]){
        uip_set.insert(p);
      }
    }
    if(uip_set.size() == 1 && (*uip_set.begin()!=conflict_lit)){ // found first UIP cut
      assert(s==*uip_set.begin());
      uip_lit = s;
      return {uip_lit, uip_cut};
    }
  }
  uip_lit = decision_lit;

  return {uip_lit, uip_cut};
}

int CDCL::_run() {
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
      if (current_conflict.decision_level == _level
          && current_conflict.decision_cut.size()==1
          && current_conflict.decision_cut[0] == current_conflict.decision_lit
          && _level > 0
        ) { //
#if defined(SATTRACE)
        std::cout << "\033[1;31m" << "unsat-decision "<< current_conflict.decision_lit << "\033[0m\n";
#endif
        //return CDCLCONFLICT;
      }
      assert(_init_pos == prop._unit_pos);
      assert(_init_n == prop._unit_n);
      if(_level > 0)
        return CDCLCONFLICT;
      else {
#if defined(SATTRACE)
        std::cout << "\033[1;31m" << "unsat DL0" << "\033[0m\n";
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
#if defined(SATTRACE)
    std::cout << "SAT\n";
#endif
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
    int rc = _run();

    if (rc == CDCLCONFLICT) { // Resolve a conflict
      prop.remove_unit();
      _level--;
      auto _hc = handle_conflict(_prop_steps, false);
      if (_hc) {
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
#if defined(SATTRACE)
        std::cout << "BD " << _level+1 << " . " << _prop_lit << "\n";
#endif
        return CDCLCONFLICT;
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

int CDCL::run(){
  int _rc;
  do {
    _level = 0;
    _rc = _run();
    if(_rc == CDCLCONFLICT){
      _level = -1;
      auto _hc = handle_conflict(0, false);
      assert(_hc.has_value());
      conflicts.push_back(*_hc);
    }
  } while (_rc == CDCLCONFLICT);
  return _rc;
}

std::optional<CDCLConflict> CDCL::handle_conflict(size_t backtrack_steps, bool first) {
  if (first) {// initialize conflict
    current_conflict = CDCLConflict{};
    current_conflict.conflict_level = _level;
    current_conflict.decision_level = _level;
    _conflict_graph_edges.clear();
    _conflict_graph_nodes.clear();
    _conflict_graph_earliest_dls.clear();
    _conflict_graph_edges.reserve(8*backtrack_steps);
    _conflict_graph_nodes.reserve(backtrack_steps);
    _conflict_graph_earliest_dls.reserve(backtrack_steps);
    prop.begin_conflict();
  }
  //size_t num_preds = 0;
  ClauseT learned_clause;
  if (_level == current_conflict.decision_level - 1) {
    if(current_conflict.decision_level > 0 && current_conflict.decision_cut.size() == 1){
      // Learning a singleton clause, so continue backtracking to level 0
      current_conflict.decision_level = 0;
    } else {
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
  }
  for (size_t i = 0; i < backtrack_steps; ++i) {
    Lit unit_lit = prop.units[prop._unit_pos - 1];
    auto u = prop.unit_prop_reverse(current_conflict.conflict_level, _level);
    if (first) {
      // at the conflict decision level, identify the earliest decision level
      // with a preceding vertex.
      if(std::get<0>(u) > 0) // disregard DL 0 until we prove the learned clause is a singleton
        current_conflict.decision_level = std::min(current_conflict.decision_level, std::get<0>(u));
      if(_uip) {// will find the iup cut from the current DL conflict graph later
        for( Lit l : prop.conflict_children){
          _conflict_graph_edges.emplace_back(unit_lit, l);
        }
        _conflict_graph_nodes.push_back(unit_lit);
        _conflict_graph_earliest_dls.push_back(std::get<0>(u));
        prop._var_dec_lev_conflict[unit_lit.idx()] = -1;
      }
      if(i == backtrack_steps - 1){
        // This was the decided variable at the conflict's decision level
        current_conflict.decision_lit = unit_lit;
        // Identify the UIP cut if required
        if(_uip) {
          //current_conflict.decision_level = prop._conflict_backtrack_lev ;
          current_conflict.decision_level =
              prop._conflict_backtrack_lev > 0 ? prop._conflict_backtrack_lev : current_conflict.conflict_level;
          std::vector<Lit> uip_cut;
          Lit uip_lit;
          std::tie(uip_lit, uip_cut) = analyze_uip_cut(
                _conflict_graph_edges, prop.conflict_lit, current_conflict.decision_lit);
          for( Lit l : uip_cut){ //
            prop._var_dec_lev_conflict[l.idx()] = current_conflict.conflict_level;
          }
          // re-evaluate the earliest decision level influencing the UIP cut or the conflict

          for( size_t j = 0; j < backtrack_steps; ++j){
            Lit l = _conflict_graph_nodes[j];
            if(prop._var_dec_lev_conflict[l.idx()] == current_conflict.conflict_level
              && _conflict_graph_earliest_dls[j] > 0
            ){
              current_conflict.decision_level = std::min(current_conflict.decision_level,
                                                         _conflict_graph_earliest_dls[j]);
            }
          }
          current_conflict.uip_lit = uip_lit;
#if defined(SATTRACE)
          std::cout << "UIP " << current_conflict.uip_lit << " " << uip_cut.size() << " -> "
                    << current_conflict.decision_level << "\n";
#endif
          current_conflict.decision_cut.push_back(uip_lit);

        } else {
          if(prop._conflict_backtrack_lev > 0)
            current_conflict.decision_level = std::min(current_conflict.decision_level, prop._conflict_backtrack_lev);
          current_conflict.decision_cut.push_back(unit_lit);
        }

      }
    }
    if (!first && std::get<1>(u) && _level > 0) {
      // This literal is the parent to a consequent in the conflict level
      current_conflict.decision_cut.push_back(unit_lit);
    }
  }

  return {};
}
