# Author: Humberto Munoz Bauza (humberto.munozbauza@nasa.gov)
#
# Copyright © 2024, United States Government, as represented by the Administrator
# of the National Aeronautics and Space Administration. All rights reserved.
#
# The PySA, a powerful tool for solving optimization problems is licensed under
# the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from itertools import chain

import numpy as np

from typing import Iterable, List, TextIO

from enum import IntFlag, auto
from collections import defaultdict
from copy import copy, deepcopy


def lvar(l):
    if l == 0:
        return None
    if l > 0:
        return l - 1
    else:
        return -l - 1


def lsign(l):
    if l == 0:
        return None
    return False if l > 0 else True


def litv(var, sign):
    """ Literal of a variable index (zero-based) and sign bit
    """
    return int(-(var + 1)) if sign else int(var+1)


class ClauseFlag(IntFlag):
    UNIT=auto()
    SAT2=auto()
    SAT=auto()
    SIMP=auto()
    UNSAT=auto()


ClfNone = ClauseFlag(0)
ClfUnit = ClauseFlag.UNIT
ClfSat2 = ClauseFlag.SAT2
ClfSat = ClauseFlag.SAT
ClfSimp = ClauseFlag.SIMP
ClfUnsat = ClauseFlag.UNSAT


class SATClause:
    def __init__(self, literals=None) -> None:
        self._lits = []
        self._flags = ClfUnsat
        if literals is not None:
            for l in literals:
                self.add_lit(l)

    @property
    def literals(self):
        return self._lits
    
    def __iter__(self):
        return iter(self._lits)

    def __getitem__(self, i):
        return self._lits[i]
    
    def __repr__(self):
        return ' '.join([str(i) for i in self._lits] + ['0'])
    
    @property
    def flags(self):
        return self._flags
    
    def size(self):
        return len(self._lits)
    
    def add_lit(self, l):
        self._lits.append(l)
        k = self.size()
        if k == 1:
            self._flags = ClfUnit
        elif k == 2:
            self._flags = ClfSat2
        else:
            self._flags = ClfNone
    
    def remove_lit(self, i: int):
        k = self.size()
        if i < k:
            if i < k-1:
                self._lits[i], self._lits[k-1] = self._lits[k-1], self._lits[i]
            k = k - 1
            if k == 2:
                self._flags = ClfSat2
            elif k == 1:
                self._flags = ClfUnit
            else:
                self._flags = ClfUnsat
            return self._lits.pop()
        else:
            return None

    def is_unit(self):
        return ClfUnit in self._flags
    
    def is_empty(self):
        return ClfUnsat in self._flags
    
    def is_binary_disj(self):
        return ClfSat2 in self._flags
    
    def mark_sat(self):
        self._flags = self._flags | ClfSat

    def is_sat(self):
        return ClfSat in self._flags
    
    def simplify(self):
        # Simplify the clause for duplicate literals and check whether
        # the clause is a tautology (l ^ ~l)
        k = self.size()
        simpl = False
        for i in range(k-1):
            for j in range(k-i-1):
                li = self._lits[k-i-1]
                lj = self._lits[j]
                if( li == lj):
                    self.remove_lit(k-i-1)
                    simpl = True
                    break
                nlj = - lj
                if li == nlj: # clause is a tautotology
                    return ClfSat
        if simpl:
            return ClfSimp
        else:
            return ClfNone
        
    def propagate_lit(self, lit):
        # Propagate the truth value of a literal to simplify the clause. 
        # If lit is in the clause, returns a SAT flags. Removes any occurrence of ~lit
        # and returns the CLSIMP flag. 
        remv = False
        k = self.size()
        for i in range(k):
            if lit == self._lits[k-i-1]:
                return ClfSat
            elif -lit == self._lits[k-i-1]:
                self.remove_lit(k-i-1)
                remv = True
        if remv:
            return ClfSimp
        else:
            return ClfNone

def clause_resolve(cl1: SATClause, cl2: SATClause, lit: int):
    """
    Apply the resolution operator on two clauses on the literal lit
    """

    cl_res = SATClause()
    
    # Remove 'lit' from cl1 and '-lit' from cl2
    for l in cl1:
        if l != lit:
            cl_res.add_lit(l)
    for l in cl2:
        if l != -lit:
            cl_res.add_lit(l)


    return cl_res

def _as_clause(cl):
    if isinstance(cl, SATClause):
        return cl
    else:
        return SATClause(cl)


class SATFormula:
    def __init__(self, clauses=None) -> None:
        self._clause_list = []
        self._n_vars = 0
        if clauses is not None:
            for cl in clauses:
                self.add_clause(cl)

    @property
    def clauses(self) -> List[SATClause]:
        return self._clause_list
    
    def num_clauses(self):
        return len(self._clause_list)
    
    def num_vars(self):
        return self._n_vars
    
    def active_vars(self):
        """
        Evaluate the set of variables involved in an unsatisfied clause
        """
        vset = defaultdict(int)
        for cl in self._clause_list:
            if not cl.is_sat():
                for l in cl:
                    v = lvar(l)
                    vset[v] += 1
        return vset
    
    def count_weights(self):
        w = defaultdict(int)
        nclauses = self.num_clauses()
        for i in range(nclauses):
            cl = self[i]
            k = cl.size()
            if not cl.is_sat():
                w[k] += 1
        return w
    
    def add_clause(self, clause):
        _cl = _as_clause(clause)
        for l in _cl:
            v = lvar(l)
            if v >= self._n_vars:
                self._n_vars = v + 1
        self._clause_list.append(_as_clause(clause))

    def sub_formula(self, clause_list):
        """ Construct a subformula from a list of clause indices.
        The subformula includes shallow copies of the clauses and the number of variables is not compressed
        """
        new_formula = SATFormula()
        for cli in clause_list:
            cl = self._clause_list[cli]
            new_formula._clause_list.append(cl)
        new_formula._n_vars = self._n_vars
        return new_formula
    
    def compress_vars(self):
        active_vars = self.active_vars()
        sorted_vars = sorted(active_vars.keys())
        map_to_compr = {v: i for (i, v) in enumerate(sorted_vars)}
        map_from_compr = {i: v for (v, i) in map_to_compr.items()}
        _compr = SATFormula()
        for cl in self._clause_list:
            _cl = SATClause()
            for l in cl:
                _v = map_to_compr[lvar(l)]
                _l = litv(_v, lsign(l))
                _cl.add_lit(_l)
            _compr.add_clause(_cl)
        assert _compr.num_vars() == len(active_vars)
        return _compr, map_to_compr, map_from_compr
    
    def __getitem__(self, i):
        return self._clause_list[i]
    
    def copy(self):
        new_formula = SATFormula()
        for cl in self._clause_list:
            new_formula._clause_list.append(copy(cl))
        new_formula._n_vars = self._n_vars

    def as_list(self):
        """
        Represent formula as a nested list of list of literals
        """
        li = []
        for cl in self.clauses:
            cl_li = []
            for l in cl.literals:
                cl_li.append(l)
            li.append(cl_li)
        return li


class SATIndexer:
    def __init__(self, clauses=None) -> None:
        self.clauses_by_lit = defaultdict(list)
        self._ncl = 0
        if clauses is not None:
            for cl in clauses:
                self.include_clause(cl)
    def include_clause(self, cl):
        cl = _as_clause(cl)
        for lit in cl.literals:
            self.clauses_by_lit[lit].append(self._ncl)
        self._ncl += 1


class SATProp:
    def __init__(self):
        self._units = []
        self.implications = []
        self.antecedents = None
        self.consequents = None
        self.formula = None
        self.sat_prop = None
        self.n_unsat = 0
        self.decided_vars = None

    def initialize(self, formula: SATFormula):
        """
        Initialize or reset the propagator with a SAT formula.
        The propagator stores a deep copy of the formula
        """
        self._units.clear()
        self.implications.clear()
        self.formula = deepcopy(formula)
        self.antecedents = -np.ones(formula.num_vars(), dtype=int)
        self.consequents = np.zeros(formula.num_clauses(), dtype=int)
        self.sat_prop = SATIndexer(formula.clauses)
        self.decided_vars = np.zeros(formula.num_vars(), dtype=np.uint8)
        for i, cl in enumerate(self.formula.clauses):
            if cl.is_unit():
                self.push_unit(cl[0])
                self.consequents[i] = cl[0]
                self.antecedents[lvar(cl[0])] =  i
                self.implications.append(('o', i, SATClause(cl)))
        self.n_unsat = 0

    def push_unit(self, unit_lit):
        self._units.append(unit_lit)
    
    def propagate_units(self):  # Generator for units in the current formula
        if self.sat_prop is None:
            raise RuntimeError("SATProp not initialized")
        
        while len(self._units) > 0:
            unit_lit = self._units.pop()
            #unit_cl = self.formula.clauses[unit_cli]

            #unit_lit = unit_cl[0]
            var = lvar(unit_lit)
            if self.decided_vars[var]:
                raise RuntimeError(f"Variable {var} already propagated")
            self.decided_vars[var] = 1
            pos_clauses = self.sat_prop.clauses_by_lit[unit_lit]
            neg_clauses = self.sat_prop.clauses_by_lit[-unit_lit]
            # Propagate literal to its positive clauses, which are now satisfied.
            # and can be deleted from the indexed
            npos = len(pos_clauses)
            nneg = len(neg_clauses)
            for cli in pos_clauses:
                cl = self.formula.clauses[cli]
                if cl.is_sat():
                    continue
                prop_flags = cl.propagate_lit(unit_lit)
                assert ClfSat in prop_flags
                cl.mark_sat()
                self.implications.append(('ud', cli, SATClause(cl)))
            # Propagate literal to its negative clauses
            for cli in neg_clauses:
                cl = self.formula.clauses[cli]
                if cl.is_sat():
                    continue
                prop_flags = cl.propagate_lit(unit_lit)
                assert ClfSimp in prop_flags
                self.implications.append(('ua', cli, SATClause(cl),
                                          'l', unit_lit
                                          ))
                if cl.is_unit():  # new unit clause found
                    self.consequents[cli] = cl[0]
                    v = lvar(cl[0])
                    # Add unit to stack if the variable does not already have an antecedent
                    if self.antecedents[v] < 0:
                        self.antecedents[v] = cli
                        self.push_unit(cl[0])
                elif cl.is_empty():
                    self.n_unsat += 1
            yield unit_lit

    def propagate(self):
        for _ in self.propagate_units():
            pass
        return self.implications

    def var_priority(self, var):
        if self.decided_vars[var]:
            return 0.0
        poscl = [self.formula.clauses[cli] for cli in self.sat_prop.clauses_by_lit[litv(var, False)]
                    if not self.formula.clauses[cli].is_sat()]
        negcl = [self.formula.clauses[cli] for cli in self.sat_prop.clauses_by_lit[litv(var, True)]
                    if not self.formula.clauses[cli].is_sat()]
        if len(poscl) == 0 or len(negcl) == 0:
            return 0.0
        klist = [cl.size() for cl in chain(poscl, negcl)]
        kmin = min(klist)
        return - kmin * np.log(len(poscl)*len(negcl))

def read_cnf(f: TextIO):
    line_num = 0
    parsed_problem_line = False
    formula = SATFormula()
    current_clause = SATClause()
    for line in f:
        line_num += 1
        if len(line) == 0:
            continue
        if line[0].lower() == 'c':
            continue
        words = line.split()
        if len(words) == 0:
            continue
        if words[0] == 'p':
            if parsed_problem_line:
                raise RuntimeError("CNF problem line parsed multiple times.")
            parsed_problem_line = True
            continue
        if not parsed_problem_line:
            raise RuntimeError("CNF file in wrong format.")
        for _istr in words:
            try: 
                i = int(_istr)
            except Exception as e:
                raise RuntimeError("Expected integer token in CNF.")
            
            if i != 0:
                current_clause.add_lit(i)
            else:
                formula.add_clause(current_clause)
                current_clause = SATClause()
    if not current_clause.is_empty():
        raise RuntimeError("Expected final CNF clause terminated by 0.")

    return formula

def write_cnf(formula: SATFormula, f: TextIO, comments=None):
    f.write(f"p cnf {formula.num_vars()} {formula.num_clauses()}\n")
    if comments is not None:
        for cstr in comments:
            f.write(f"c {cstr}\n")
    for cl in formula.clauses:
        f.write(f"{cl}\n")
