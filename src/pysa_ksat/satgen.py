import argparse

import numpy as np
import hashlib

import matplotlib.pyplot as plt
from scipy.stats import gamma, skew, lognorm, kurtosistest, normaltest, expon, weibull_min
from tqdm import tqdm
from typing import List, Optional, Union

from .util.sat import litv, lvar, SATFormula, SATProp, SATClause
from .random_ksat import KSatGenerator
# from pysa2.models.sat import SatK
# from pysa2.generators.sat.ksat import KSatGenerator

from pysa2_cdcl.bindings import cdcl_optimize, _CDCLResult
from pysa_dpll.sat import optimize as dpll_optimize
from pysa_walksat.bindings import walksat_optimize


class SatGen:
    def __init__(self, n, m, k=3):
        self._n = n
        self._m = m
        self._k = k
        self._instances  = []
        self._solutions = []
        self._backbones = None
        self._entailed_literals = None
        self._last_sat_fraction = None
        self._instance_sha = None

    def add_instances(self, generator, num_insts, max_tries=None, filter_sat=True, rng: Optional[Union[int, np.random.Generator]]=None):
        """
        Add SAT instances from a generator, optionally filtering by solvable instances
        """
        if filter_sat:
            new_instances = []
            new_sols = []
            with tqdm(total=num_insts) as pbar:
                actual_tries = max_tries
                for i in range(max_tries):
                    _clauses = generator.generate(rng=rng)
                    _res = cdcl_optimize(_clauses, True)
                    if _res.result_state is not None:
                        new_instances.append(_clauses)
                        new_sols.append([np.asarray(_res.result_state, dtype=np.int8)])
                        pbar.update(1)
                    if len(new_instances) == num_insts:
                        actual_tries = i + 1
                        break
                else:
                    print(f"Maximum number of tries reached: {len(new_instances)} instances in {max_tries} tries.")
                self._last_sat_fraction = len(new_instances) / actual_tries
                print(f"Observed SAT fraction: {self._last_sat_fraction}")

        else:
            new_instances = list(generator.generate_n(num_insts))
            new_sols = [[]]*num_insts
        self._instances += new_instances
        self._solutions += new_sols


    def evaluate_backbones(self):
        self._backbones = []
        self._entailed_literals = []
        for i in tqdm(range(len(self._instances))):
            _inst0 = self._instances[i]
            _sol0 = self._solutions[i][0]
            _entailed_lits = []
            bb = 0
            for j, sj in enumerate(_sol0):
                lj = litv(j, sj==0)
                _res = cdcl_optimize(_inst0.clauses + [[-lj]])
                if _res.result_state is None:
                    _entailed_lits.append(lj)
                    bb += 1
            self._backbones.append(bb)
            self._entailed_literals.append(_entailed_lits)

    def enumerate_solutions(self, full_dpll=False):
        """
        Enumerate all solutions of each instance with DPLL.
        If backbone information is available, it will be used to reduce the number of variables before running DPLL.
        Otherwise,
        """
        #opt_dpll = DPLLOptimizer(max_n_unsat=0, n_threads=None)

        if self._backbones is not None and not full_dpll:
            for i in tqdm(range(len(self._instances))):
                self._solutions[i] = []
                n = self._instances[i].nvars()
                nclauses = self._instances[i].nclauses()
                formula = SATFormula(self._instances[i].clauses + [[l] for l in self._entailed_literals[i]])
                base_solution = np.zeros(n, dtype=np.int8)
                entailed_vars =set([ lvar(l) for l in self._entailed_literals[i]])
                for l in self._entailed_literals[i]:
                    if l > 0:
                        base_solution[lvar(l)] = 1
                prop = SATProp()
                prop.initialize(formula)
                prop.propagate()
                subf = prop.formula.sub_formula( [i for i in range(nclauses) if not prop.formula[i].is_sat()])
                compf, map_to_cf, map_from_cf = subf.compress_vars()
                assert(set(map_from_cf.values()).isdisjoint(entailed_vars))
                dpll_solutions, _  = dpll_optimize(compf.clauses, max_n_unsat=0, n_threads=None)
                for config in dpll_solutions:
                    _sol = np.copy(base_solution)
                    for j in range(len(map_to_cf)):
                        _sol[map_from_cf[j]] = config[j]
                    self._solutions[i].append(_sol)

        else:
            for i in tqdm(range(len(self._instances))):
                _inst = self._instances[i]
                dpll_solutions, _ = dpll_optimize(_inst, max_n_unsat=0, n_threads=None)
                dpll_configs = list(np.asarray(res) for res in dpll_solutions)
                self._solutions[i] = dpll_configs


        # SHA digest for debugging
        m = hashlib.sha256()
        for _sol in self._solutions:
            _solstrs = sorted([''.join([str(c) for c in _arr]) for _arr in _sol])
            for _s in _solstrs:
                m.update(_s.encode())
        print(f"enumerate_solutions() SHA256: {m.hexdigest()}")


    def instance_sha(self, i):
        m = hashlib.sha256()
        _inst = self._instances[i]
        for cl in _inst:
            m.update(f"{SATClause(cl)}\n".encode())
        if self._solutions is not None:
            _solstrs = sorted([''.join([str(c) for c in _arr]) + '\n' for _arr in self._solutions[i]])
            for _s in _solstrs:
                m.update(_s.encode())
        shadigest = m.hexdigest()
        return shadigest
