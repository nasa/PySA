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

import ctypes
import scipy
import numpy as np
from typing import List, NoReturn, Tuple, Any, Callable, Optional
import pandas as pd
from time import time
import numba
from tqdm.auto import tqdm
import pysa.utils as utils

Vector = List[float]
Matrix = List[List[float]]
State = List[float]
RefProbFun = Callable[[Vector, Optional[int]], float]


@numba.njit(fastmath=True, nogil=True, parallel=False)
def get_energy(couplings: Matrix, local_fields: Vector, state: State) -> float:
    """
  Compute energy given couplings and local fields.
  """
    return state.dot(couplings.dot(state) / 2 + local_fields)


@numba.njit(fastmath=True, nogil=True, parallel=False)
def update_spin(couplings: Matrix, local_fields: Vector, state: State, pos: int,
                beta: float, log_r: float) -> float:
    """
  Update spin accordingly to Metropolis update.
  """

    # Get the negate delta energy (qubo)
    delta_n_energy = (2. * state[pos] - 1.) * (couplings[pos].dot(state) +
                                               local_fields[pos])

    # Metropolis update
    if delta_n_energy >= 0 or log_r < beta * delta_n_energy:

        # Update spin (qubo)
        state[pos] = 0 if state[pos] else 1

        # Return delta energy
        return -delta_n_energy

    else:

        # Otherwise, return no change in energy
        return 0.


@numba.njit(fastmath=True, nogil=True, parallel=False)
def update_spin_ref(couplings: Matrix, local_fields: Vector, state: State,
                    pos: int, beta: float, log_r: float,
                    ref_prob: RefProbFun) -> float:
    ''' Update spin accordingly to Metropolis update.  
    This version works based off reference distribution p_0 given by ref_prob, 
    then the weight will be (p_t**beta)*(p_0**(1-beta))'''

    # Get the negate delta energy (qubo)
    delta_n_energy = (2. * state[pos] - 1.) * (couplings[pos].dot(state) +
                                               local_fields[pos])

    ref_delta_n_energy = np.log(ref_prob(state, pos) / ref_prob(state))

    full_delta = beta * delta_n_energy + (1 - beta) * ref_delta_n_energy

    # Metropolis update
    if full_delta >= 0 or log_r < full_delta:

        # Update spin (qubo)
        state[pos] = 0 if state[pos] else 1

        return delta_n_energy
    else:

        # Otherwise, return no change in energy
        return 0.
