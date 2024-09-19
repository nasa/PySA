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
from pysa.ais import get_log_omega

Vector = List[float]
Matrix = List[List[float]]
State = List[float]
RefProbFun = Callable[[Vector, Optional[int]], float]
EnergyFunction = Callable[[Matrix, Vector, State], float]
UpdateSpinFunRef = Callable[
    [Matrix, Vector, Vector, int, float, float, RefProbFun], float]
UpdateSpinFunction = Callable[[Matrix, Vector, Vector, int, float, float],
                              float]
SweepFunction = Callable[[UpdateSpinFunction, Matrix, Vector, Vector, float],
                         float]


@numba.njit(fastmath=True, nogil=True, parallel=False)
def random_sweep(update_spin: UpdateSpinFunction, couplings: Matrix,
                 local_fields: Vector, state: State, beta: float) -> float:
    """
  Metropolis update.
  """

    # Get random numbers
    log_r = np.log(np.random.random(size=len(state)))

    # Try to update every spin
    delta_energy = 0.
    for pos in np.random.permutation(len(state)):
        delta_energy += update_spin(couplings, local_fields, state, pos, beta,
                                    log_r[pos])

    return delta_energy


@numba.njit(fastmath=True, nogil=True, parallel=False)
def sequential_sweep(update_spin, couplings: Matrix, local_fields: Vector,
                     state: State, beta: float) -> float:
    """
  Metropolis update.
  """

    # Get random numbers
    log_r = np.log(np.random.random(size=len(state)))

    # Try to update every spin
    delta_energy = 0.
    for pos in range(len(state)):
        delta_energy += update_spin(couplings, local_fields, state, pos, beta,
                                    log_r[pos])

    return delta_energy


@numba.njit(fastmath=True, nogil=True, parallel=False)
def random_sweep_ref(update_spin: UpdateSpinFunRef, couplings: Matrix,
                     local_fields: Vector, state: State, beta: float,
                     ref_prob: RefProbFun) -> float:
    '''This sweeps through metropolis updates for the entire State state
    proposing a flip to each bit in a randomized order.  This version is
    designed for Annealed Importance sampling and takes in the reference 
    probability distribution.  It returns the change to the log of the
    Boltzmann weight'''

    # Get random numbers
    log_r = np.log(np.random.random(size=len(state)))

    # Try to update every spin
    delta_energy = 0.
    for pos in np.random.permutation(len(state)):
        delta_energy += update_spin(couplings, local_fields, state, pos, beta,
                                    log_r[pos], ref_prob)

    return delta_energy


@numba.njit(fastmath=True, nogil=True, parallel=False)
def sequential_sweep_ref(update_spin: UpdateSpinFunRef, couplings: Matrix,
                         local_fields: Vector, state: State, beta: float,
                         ref_prob: RefProbFun) -> float:
    '''This sweeps through metropolis updates for the entire State state
    proposing a flip to each bit in a sequential order.  This version is
    designed for Annealed Importance sampling and takes in the reference 
    probability distribution.  It returns the change to the log of the
    Boltzmann weight'''

    # Get random numbers
    log_r = np.log(np.random.random(size=len(state)))

    # Try to update every spin
    delta_energy = 0.
    for pos in range(len(state)):
        delta_energy += update_spin(couplings, local_fields, state, pos, beta,
                                    log_r[pos], ref_prob)

    return delta_energy


@numba.njit(fastmath=True, nogil=True, parallel=True)
def simulation_parallel(update_spin: UpdateSpinFunction,
                        sweep: SweepFunction,
                        couplings: Matrix,
                        local_fields: Vector,
                        states: List[State],
                        energies: List[float],
                        beta_idx: List[int],
                        betas: List[float],
                        n_sweeps: int,
                        get_part_fun: bool = False,
                        use_pt: bool = True) -> Tuple[State, float, int, int]:
    """
  Apply simulation.
  """

    # Get number of replicas
    n_replicas = len(states)

    # Best configuration/energy
    _best_energy = np.copy(energies)
    _best_state = np.copy(states)
    _best_sweeps = np.zeros(n_replicas, dtype=np.int32)
    betas_sorted = np.empty_like(betas)
    log_omegas = np.zeros(n_sweeps)

    # For each run ...
    for s in range(n_sweeps):
        for k in range(n_replicas):
            betas_sorted[beta_idx[k]] = betas[k]
        # ... apply sweep for each replica ...
        for k in numba.prange(n_replicas):

            # Apply sweep
            energies[k] += sweep(update_spin, couplings, local_fields,
                                 states[k], betas_sorted[k])

            # Store best state
            if energies[k] < _best_energy[k]:
                _best_energy[k] = energies[k]
                _best_state[k] = np.copy(states[k])
                _best_sweeps[k] = s

        # ... and pt move.
        if use_pt:
            utils.pt(states, energies, beta_idx, betas)
        # Calculate the weights for the partition function
        if get_part_fun:
            log_omegas[s] = get_log_omega(betas, beta_idx, energies)

    # Get lowest energy
    best_pos = np.argmin(_best_energy)
    best_state = _best_state[best_pos]
    best_energy = _best_energy[best_pos]
    best_sweeps = _best_sweeps[best_pos]

    # Return states and energies
    return ((states, energies, beta_idx, log_omegas), (best_state, best_energy,
                                                       best_sweeps, s + 1))


@numba.njit(fastmath=True, nogil=True, parallel=False)
def simulation_sequential(update_spin: UpdateSpinFunction,
                          sweep: SweepFunction,
                          couplings: Matrix,
                          local_fields: Vector,
                          states: List[State],
                          energies: List[float],
                          beta_idx: List[int],
                          betas: List[float],
                          n_sweeps: int,
                          get_part_fun: bool = False,
                          use_pt: bool = True) -> Tuple[State, float, int, int]:
    """
  Apply simulation.
  """

    # Get number of replicas
    n_replicas = len(states)

    # Fix initial bests
    best_state = states[0]
    best_energy = energies[0]
    best_sweeps = 0
    betas_sorted = np.empty_like(betas)
    log_omegas = np.zeros(n_sweeps)

    # For each run ...
    for s in range(n_sweeps):
        for k in range(n_replicas):
            betas_sorted[beta_idx[k]] = betas[k]
        # ... apply sweep for each replica ...
        for k in range(n_replicas):

            # Apply sweep
            energies[k] += sweep(update_spin, couplings, local_fields,
                                 states[k], betas_sorted[k])

            # Store best state
            if energies[k] < best_energy:
                best_energy = energies[k]
                best_state = np.copy(states[k])
                best_sweeps = s

        # ... and pt move.
        if use_pt:
            utils.pt(states, energies, beta_idx, betas)
        # Calculate the weights for the partition function
        if get_part_fun:
            log_omegas[s] = get_log_omega(betas, beta_idx, energies)

    # Return states and energies
    return ((states, energies, beta_idx, log_omegas), (best_state, best_energy,
                                                       best_sweeps, s + 1))


@numba.njit(fastmath=True, nogil=True, parallel=False)
def ais_simulation_sequential(temps: Vector, update_spin: UpdateSpinFunction,
                              sweep_func: SweepFunction, ref_prob: RefProbFun,
                              energy_fun: EnergyFunction, couplings: Matrix,
                              local_fields: Vector, initial_states: Matrix):
    '''Goes through a full sequence of Annealed Importance Sampling
    steps.  This will output both the samples collected as well as the 
    log of the weights'''
    num_samps = len(initial_states)

    samples = np.copy(initial_states)
    log_omegas = np.zeros(num_samps)
    # Go through and take all the samples
    for i in range(num_samps):  # This loop is trivially parallelizable

        # Initialize the state based on the reference probability distribution
        state = initial_states[i]

        sample, log_omega = ais_sample(temps, state, update_spin, sweep_func,
                                       ref_prob, energy_fun, couplings,
                                       local_fields)
        samples[i] = sample
        log_omegas[i] = log_omega

    return samples, log_omegas


@numba.njit(fastmath=True, nogil=True, parallel=True)
def ais_simulation_parallel(temps: Vector, update_spin: UpdateSpinFunction,
                            sweep_func: SweepFunction, ref_prob: RefProbFun,
                            energy_fun: EnergyFunction, couplings: Matrix,
                            local_fields: Vector, initial_states: Matrix):
    '''Goes through a full sequence of Annealed Importance Sampling
    steps.  This will output both the samples collected as well as the 
    log of the weights'''
    num_samps = len(initial_states)

    samples = np.copy(initial_states)
    log_omegas = np.zeros(num_samps)
    # Go through and take all the samples
    for i in numba.prange(num_samps):  # This loop is trivially parallelizable

        # Initialize the state based on the reference probability distribution
        state = initial_states[i]

        sample, log_omega = ais_sample(temps, state, update_spin, sweep_func,
                                       ref_prob, energy_fun, couplings,
                                       local_fields)
        samples[i] = sample
        log_omegas[i] = log_omega

    return samples, log_omegas


@numba.njit(fastmath=True, nogil=True, parallel=False)
def ais_sample(temps: Vector, state: State, update_spin: UpdateSpinFunction,
               sweep_func: SweepFunction, ref_prob: RefProbFun,
               energy_fun: EnergyFunction, couplings: Matrix,
               local_fields: Vector):
    '''This runs annealed importance sampling once and will output the
    final sample as well as the log of the (unnormalized) weight 
    associated with that sample'''

    # run the actual sweeps
    log_omega = 0  # this is the log of the unnormalized AIS weight
    beta_old = 0
    for beta in temps:
        weight = sweep_func(update_spin, couplings, local_fields, state, beta,
                            ref_prob)
        # Calculate the current energies
        cur_energy = energy_fun(couplings, local_fields, state)
        ref_energy = -np.log(ref_prob(state))
        # Calculate the log of the AIS weight
        log_pk_xk = -beta * cur_energy - (1 - beta) * ref_energy
        log_pk_xkm = -beta_old * cur_energy - (1 - beta_old) * ref_energy
        log_omega += log_pk_xk - log_pk_xkm
        # remember what beta was for the last step
        beta_old = beta
    return state, log_omega
