"""
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
"""

from typing import List, NoReturn, Tuple, Any
import numpy as np
import scipy
import numba

Vector = List[float]
Matrix = List[List[float]]
State = List[float]


@numba.njit(cache=True, fastmath=True, nogil=True, parallel=False)
def pt(states: List[State], energies: List[float],
       betas: List[float]) -> NoReturn:
    """
  Parallel tempering move.
  """

    # Get number of replicas
    n_replicas = len(states)

    # Apply PT for each pair of replicas
    for k in range(n_replicas - 1):

        # Get first index
        k1 = n_replicas - k - 1

        # Get second index
        k2 = n_replicas - k - 2

        # Compute delta energy
        de = (energies[k1] - energies[k2]) * (betas[k1] - betas[k2])

        # Accept/reject following Metropolis
        if de >= 0 or np.random.random() < np.exp(de):
            betas[k1], betas[k2] = betas[k2], betas[k1]


def get_problem_matrix(instance: List[Tuple[int, int, float]]) -> Matrix:
    """
    Generate problem matrix from a list of interactions.
    """

    from more_itertools import flatten
    from collections import Counter

    # Check that couplings are not repeated
    if set(Counter(tuple(sorted(x[:2])) for x in instance).values()) != {1}:
        warn("Duplicated couplings are ignored.")

    # Get list of variables
    _vars = sorted(set(flatten(x[:2] for x in instance)))

    # Get number of variables
    _n_vars = len(_vars)

    # Get map
    _map_vars = {x: i for i, x in enumerate(_vars)}

    # Get problem
    problem = np.zeros((_n_vars, _n_vars))
    for x, y, J in instance:
        x = _map_vars[x]
        y = _map_vars[y]
        problem[y][x] = problem[x][y] = J

    return problem
