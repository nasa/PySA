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

# This code implements Annealed Importance Sampling: arXiv:physics/9803008
# for implementation with the rest of pysa
# The original version of this code was written by Lucas Brady

import numpy as np
from typing import List
import numba
from time import time
import pandas as pd

Vector = List[float]


def partition_function_post(solution: pd.DataFrame):
    '''Takes in a pandas DataFrame that might result from a metropolis_update
    from the PySA Solver and gives you an estimate of what the partition
    function of the highest beta value (lowest temperature) distribution
    is.  Note that this will improve in accuracy the larger num_replicas
    and num_reads was.  This function also requires that one of the
    beta values used by the annealer was 0'''

    # Extract information from the DataFrame
    n = len(solution["states"][0][0])  # number of bits
    temps = solution["temps"]
    energies = solution["energies"]
    num_samps = len(temps)

    log_omegas = []  # This will hold the logs of the AIS weights
    for s in range(num_samps):

        # Get the info for the current sample
        cur_temps = temps[s]
        cur_betas = 1 / cur_temps
        cur_energies = energies[s]

        log_omegas += [get_log_omega(cur_betas, cur_energies)]

    logZ0 = n * np.log(2)  # assuming the uniform distribution with all
    #unnormalized probabilities set to one
    logZf = omegas_to_partition(np.array(log_omegas), logZ0)
    return logZf


def combine_logZf(solution: pd.DataFrame):
    '''Takes in a pandas DataFrame that might result from a metropolis_update
    from the PySA Solver and gives you an estimate of the log of the partition 
    function Zf taking into account all the runs combined
    This version works only on DataFrames produced with 
    the flag get_part_fun = True'''

    log_Zf = solution['log_Zf'].to_numpy()

    Ns = len(log_Zf)  # the number of samples
    smax = log_Zf.max()

    expsum = 0
    for i in range(Ns):
        s = log_Zf[i] - smax
        expsum += np.exp(s)

    return smax - np.log(Ns) + np.log(expsum)


@numba.njit(fastmath=True, nogil=True, parallel=False)
def get_log_omega(betas: Vector, beta_idx: List[int], energies: Vector):
    '''Takes in the temperatures and energies of an annealing
    procedure and outputs what log_omega the AIS weight used for
    calculating the log of the partition function'''

    num_replicas = len(betas)

    # Argsort temperatures
    _sort = np.argsort(betas)

    # Sort temperatures and states
    cur_betas = betas[_sort]
    cur_energies = energies[beta_idx[_sort]]

    if cur_betas[0] != 0:
        raise ValueError(
            '''Calculating the Partition Function requires a zero beta.  The most likely solution is that when you are calling metrpolis_update you need to include np.inf in your temperature list (if using user defined temperatures) or use the beta0 = True option (if using min_temp and max_temp).'''
        )

    log_omega = 0
    beta_old = 0

    for r in range(1, num_replicas):
        beta = cur_betas[r]
        energy = cur_energies[r]

        # Calculate the log of the AIS weight
        log_pk_xk = -beta * energy
        log_pk_xkm = -beta_old * energy
        log_omega += log_pk_xk - log_pk_xkm

        beta_old = beta
    return log_omega


@numba.njit(fastmath=True, nogil=True, parallel=False)
def omegas_to_partition(log_omegas: Vector, logZ0: float):
    '''Takes in a list of log omegas which are the unnormalized cumulative
    weights obtained from annealed importance sampling and logZ0, the partition
    function from the reference probability distribution.  Returns an
    estimate of the log of the partition function Zf of the final annealed 
    importance sampling distribution'''

    Ns = len(log_omegas)  # the number of samples
    smax = log_omegas.max()

    expsum = 0
    for i in range(Ns):
        s = log_omegas[i] - smax
        expsum += np.exp(s)

    return smax - np.log(Ns) + logZ0 + np.log(expsum)


########### This set of functions is associated with "uniform"


def uniform_prob_initialization(n: int, problem_type: str, initial_args=None):
    '''Randomly initializes a single state of length n.  This returns a 
    vector representing the randomly initialized state
    Currently supports problem_type: "ising" -> randomly chooses \pm 1
                                     "qubo" -> randomly chooses 0 or 1'''
    if problem_type == "qubo":
        state = np.random.randint(2, size=n)
    elif problem_type == "ising":
        state = 2 * np.random.randint(2, size=n) - 1
    else:
        raise ValueError(
            f"self.problem_type=='{self.problem_type}' not supported.")
    return np.array(state)


@numba.njit(fastmath=True, nogil=True, parallel=False)
def uniform_ref_prob(state: Vector, pos: int = None):
    '''Takes in a state vector and returns its probability under a uniform
    probability distribution.  Note that this gives the unnormalized
    probability, and the normalization factor is given by
    uniform_partition_fun
    pos is the position in the state that has been proposed for a change
    In the uniform reference probability, pos and state do nothing'''
    return 1  # every state is equally likely


@numba.njit(fastmath=True, nogil=True, parallel=False)
def uniform_partition_fun(n: int):
    '''returns the log of the partition function for a uniform probability 
    distribution this function is designed to be consistent with the 
    normalization in uniform_ref_prob which is itself unnormalized'''

    return n * np.log(2)


########### This set of functions is associated with "bernoulli"


def bernoulli_prob_initialization(n: int,
                                  problem_type: str,
                                  initial_args=[0.5]):
    '''Randomly initializes a single state of length n.  This returns a 
    vector representing the randomly initialized state
    Currently supports problem_type: "ising" -> randomly chooses \pm 1
                                     "qubo" -> randomly chooses 0 or 1
    This generates based off a Bernoulli distribution.  The probability of
    a -1 or 0 (ising/qubo) is given by the zeroth element of initial_args'''
    p = initial_args[0]

    if problem_type == "qubo":
        state = [np.random.binomial(1, 1 - p) for i in range(n)]

    elif problem_type == "ising":
        state = [2 * np.random.binomial(1, 1 - p) - 1 for i in range(n)]

    else:
        raise ValueError(
            f"self.problem_type=='{self.problem_type}' not supported.")
    return np.array(state)


@numba.njit(fastmath=True, nogil=True, parallel=False)
def bernoulli_ref_prob(state: Vector, p: float, pos: int = -1):
    '''Takes in a state vector and returns its probability under a Bernoulli
    probability distribution.  Note that this gives the unnormalized
    probability, and the normalization factor is given by
    bernoulli_partition_fun
    pos is the position in the state that has been proposed for a change'''
    n = len(state)
    c = 0
    for b in range(n):
        if b == pos:
            if state[b] == 1:
                bit = 0
            else:
                bit = 1
        else:
            bit = state[b]
        if bit == 1:
            c += 1
    return (p**(n - c)) * ((1 - p)**c)


@numba.njit(fastmath=True, nogil=True, parallel=False)
def bernoulli_partition_fun(n: int):
    '''returns the log of the partition function for a bernoulli probability 
    distribution this function is designed to be consistent with the 
    normalization in bernoulli_ref_prob which is itself actually normalized'''

    return 0
