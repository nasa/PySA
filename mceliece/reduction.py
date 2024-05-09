# Author: Humberto Munoz Bauza (humberto.munozbauza@nasa.gov)
#
# Copyright © 2023, United States Government, as represented by the Administrator
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

from copy import copy

import numpy as np
from numba.typed import typeddict
from typing import Dict, Tuple, List
from pysat.formula import CNF


def xorsat_reduction_step(xorsat_clauses: List[List[int]],
                          aux_clauses: Dict[Tuple[int, int], int],
                          next_aux_var: int) -> List[List[int]]:
    new_clauses = []
    m = len(xorsat_clauses)
    # xor_equivalences = np.zeros((m, n-1, 3), dtype=np.int64)
    aux_var = next_aux_var
    for i in range(m):
        Ci = xorsat_clauses[i]
        new_Ci = []
        ki = len(Ci)
        if ki > 2:
            for j in range(ki // 2):
                u = Ci[2 * j]
                v = Ci[2 * j + 1]
                if (u, v) in aux_clauses:
                    new_Ci.append(aux_clauses[(u, v)])
                else:
                    aux_clauses[(u, v)] = aux_var
                    new_Ci.append(aux_var)
                    aux_var += 1
            if ki % 2 == 1:
                new_Ci.append(Ci[-1])
            new_clauses.append(new_Ci)
        else:
            new_clauses.append(Ci)
    return new_clauses, aux_var


def to_xorsat(G, y=None):
    """
    Creates an xor-sat instance for the GF(2) inversion problem
        y = G x
    where G is a (m, n) matrix (n variables, m clauses)
    :param G:
    :param y:
    :return a list of length n of numpy arrays. Each array represents a clause as a sequence of up to k integers
    between -n and n, except zero. A negative integer represents a negated variable
    """
    if y is not None:
        if y.shape[0] != G.shape[0]:
            raise ValueError(f"Invalid input shapes for G and y")
    Gnz = [1 + np.asarray(np.nonzero(Gi)[0], dtype=np.int64) for Gi in G]
    if y is not None:
        for i, yi in enumerate(y):
            if yi == 0:  # clause is false, so negate one variable
                if len(Gnz[i]) > 0:
                    Gnz[i][0] *= -1
    return Gnz


def cnf_formula(reduced_xor: np.ndarray, aux_clauses: Dict[Tuple[int, int],
                                                           int]):
    sat_clauses = []
    for ci in reduced_xor:
        sat_clauses.append([ci[0], ci[1]])
        sat_clauses.append([-ci[0], -ci[1]])

    for (x1, x2), y in aux_clauses.items():
        sat_clauses.append([x1, x2, -y])
        sat_clauses.append([x1, -x2, y])
        sat_clauses.append([-x1, x2, y])
        sat_clauses.append([-x1, -x2, -y])

    return sat_clauses


class P3Ising:

    def __init__(self,
                 reduced_xor: np.ndarray,
                 aux_clauses: Dict[Tuple[int, int], int],
                 penalty=2.0):
        """
           An auxiliary variables $y == x1 \oplus x2$ can also be enforced by a weighted XOR-SAT clause
           $y \oplus x1 \oplus x2$. That is, a p=3 spin glass with ferromagnetic constraints.
           Such a Hamiltonian is eventually reduced to the sum of the problem Hamiltonian replaced with auxiliary variables
           $H_Prob = \sum_{i=1}^m ( 1 - J_i u_{i} v_{i})$
           and the 3-body constraints to enforce the value of the auxiliary variables,
           $H_constr = \sum_{c} (1 - u_c v_c w_c). $
           We have $H = A H_prob + B H_constr$
           To ensure the ground state has no constraint violations, we require B > A.
           :return:
           """
        num_aux = len(aux_clauses)
        sh = reduced_xor.shape
        m = sh[0]
        self.offset = float(m) + num_aux * penalty
        self.lin = []
        self.quad = []
        self.cub = []
        for ci in reduced_xor:
            j = 1.0
            if ci[0] < 0:
                j *= -1
            if ci[1] < 0:
                j *= -1
            if ci[1] == 0:
                self.lin.append((abs(ci[0]) - 1, j))
            else:
                self.quad.append((abs(ci[0]) - 1, abs(ci[1]) - 1, j))
        for (x1, x2), y in aux_clauses.items():
            assert x1 > 0 and x2 > 0
            self.cub.append((x1 - 1, x2 - 1, y - 1, -penalty))


class CNFXorSat:
    """
    Converts a (n, m) xor-sat instance to a sat instance in CNF.
    A maximum of m(n-1) auxiliary variables are introduced to reduce the formula to 2-xor clauses, which in turn
    are equivalent to 2-sat clauses
        x1 \\oplus x2 == (x1 | x2) & (~x1 | ~x2)

    If a clause is an xor of k literals, it will add (k-2) auxiliary variables.
    Each auxiliary variable definition introduces 4 clauses:
    $y == x1 \oplus x2$ is equivalent to the 3-SAT formula
        (x1 | x2 | ~y) & (x1 | ~x2 | y) & (~x1 | x2 | y) & (~x1 | ~x2 | ~y)
    """

    def __init__(self, n, xorclauses, y, balanced=True):
        m = len(xorclauses)
        reduced_clauses = np.zeros((m, 2), dtype=np.int64)
        #xor_equivalences = np.zeros((m, n-1, 3), dtype=np.int64)
        aux_var = (1 + n)
        aux_tbl = {}
        if balanced:
            new_clauses = copy(xorclauses)
            max_p = max(len(Ci) for Ci in xorclauses)
            aux_var = (1 + n)
            while max_p > 2:
                new_clauses, aux_var = xorsat_reduction_step(
                    new_clauses, aux_tbl, aux_var)
                max_p = max(len(Ci) for Ci in new_clauses)
            for i in range(m):
                yi = y[i]
                Ci = new_clauses[i]
                if len(Ci) > 0:
                    reduced_clauses[i, 0] = (-Ci[0] if yi == 0 else Ci[0])
                    if len(Ci) > 1:
                        # clause is reduced to u xor v
                        reduced_clauses[i, 1] = Ci[1]
                else:
                    raise RuntimeError("Empty clause found. "
                        "Try a smaller error-correcting distance to avoid this.")
        else:
            for i in range(m):
                yi = y[i]
                Ci = xorclauses[i]
                ki = len(Ci)
                if ki > 1:
                    u = Ci[0]
                    v = Ci[1]
                    #aux_var = (1+n) + i*(n-1)
                    if ki > 2:
                        for j in range(ki - 2):
                            if (u, v) in aux_tbl:
                                u = aux_tbl[(u, v)]
                                v = Ci[j + 2]
                            else:
                                aux_tbl[(u, v)] = aux_var
                                u = aux_var
                                v = Ci[j + 2]
                                aux_var += 1
                    # clause is reduced to u xor v
                    reduced_clauses[i, 0] = (-u if yi == 0 else u)
                    reduced_clauses[i, 1] = v
                else:
                    reduced_clauses[i, 0] = (-Ci[0] if yi == 0 else Ci[0])

        self.reduced_clauses = reduced_clauses
        self.aux_tbl = aux_tbl
        cnf_clauses = cnf_formula(reduced_clauses, aux_tbl)
        self.cnf = CNF(from_clauses=cnf_clauses)

    def to_p3_xorsat(self):
        """
        An auxiliary variables $y == x1 \oplus x2$ can also be enforced by a single XOR-SAT clause
        $y \oplus x1 \oplus x2$.
        :return:
        """
        return P3Ising(self.reduced_clauses, self.aux_tbl, 2.0)
