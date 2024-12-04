import numpy as np

from .util.sat import SATFormula, SATClause, litv
from typing import Optional, Union


class KSatGenerator:
    """
    Generator for random k-SAT instances.
    Each clause samples k out of n variables, without replacement
    with each literal having a random sign.
    """
    def __init__(self, k, n, m):
        """
         k: Clause size
         n: Number of variables
         m: Number of clauses
        """
        self.k = k
        self.n = n
        self.m = m
        if n < k:
            raise RuntimeError(f"n must be grater than k. Got n={n} and k={k}.")

    def generate(self, rng:Optional[Union[int, np.random.Generator]]=None):
        _rng = np.random.default_rng(rng)
        vars = np.arange(0, self.n, dtype=int)
        formula = []

        signs = _rng.integers(0, 2, (self.m, self.k), dtype=np.int8)
        for i in range(self.m):
            _sarr = signs[i, :]
            _varr = _rng.choice(vars, self.k, replace=False)
            formula.append([litv(_v, _s > 0) for _v, _s in zip(_varr, _sarr)])
        return formula

    def generate_n(self, n, rng:Optional[Union[int, np.random.Generator]]=None):
        for _ in range(n):
            yield self.generate(rng=rng)