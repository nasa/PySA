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

from typing import NoReturn, Tuple, List, Any
from more_itertools import distribute
from itertools import repeat
from multiprocessing import Pool
from os import cpu_count
import numpy as np

import pysa.simulation as simulation
import pysa.ising as ising
import pysa.utils as utils
import pytest
from tqdm.auto import tqdm

Vector = List[float]
Matrix = List[List[float]]
State = List[float]

# Define general type
dtype = 'float'


def _get_energy(w):

    # Get objects
    states, couplings, local_fields, verbose = w

    # Get number of variables
    n_vars = len(couplings)

    # Initialize minimum energy
    min_energy = np.inf

    # Check all possible combinations
    with tqdm(states, total=verbose, disable=not verbose) as pbar:

        for state in pbar:

            # Transform state
            state = 2 * np.array([int(x) for x in bin(state)[2:].zfill(n_vars)],
                                 dtype=dtype) - 1

            # Get energy for the state
            energy = ising.get_energy(couplings, local_fields, state)

            # Store only the minimum energy
            if energy < min_energy:
                min_energy = energy

    # Return the minimum energy
    return min_energy


def get_min_energy(couplings: Matrix, local_fields: Vector):

    # Get number of variables
    n_vars = len(couplings)

    # Find minimum energy by bruteforce
    with Pool(cpu_count()) as pool:
        min_energy = min(
            pool.map(
                _get_energy,
                zip(distribute(cpu_count(),
                               range(2**n_vars)), repeat(couplings),
                    repeat(local_fields),
                    [2**n_vars // cpu_count()] + [0] * cpu_count())))

    return min_energy


def gen_random_problem(n_vars: int,
                       dtype: Any = 'float') -> Tuple[Matrix, Vector]:

    # Generate random problem
    couplings = 2 * np.random.random((n_vars, n_vars)).astype(dtype) - 1
    couplings = (couplings + couplings.T) / 2

    # Split in couplings and local fields
    local_fields = np.copy(np.diagonal(couplings))
    np.fill_diagonal(couplings, 0)

    return couplings, local_fields


@pytest.mark.parametrize(
    'n_vars', [n_vars for n_vars in range(32, 129, 32) for _ in range(20)])
def test_update_spin_ising(n_vars: int):

    # Generate random problem
    couplings, local_fields = gen_random_problem(n_vars, dtype=dtype)

    # Get initial state
    state = 2 * np.random.randint(2, size=n_vars).astype(dtype) - 1
    state_copy = np.copy(state)

    # Get energy
    energy = ising.get_energy(couplings, local_fields, state)

    for pos in np.random.permutation(n_vars):

        # Update spin (infinite temperature)
        de = ising.update_spin(couplings, local_fields, state, pos, 0, -1)

        # Update must happen
        assert (not np.allclose(state, state_copy))

        # Check updated energy
        assert (np.isclose(energy + de,
                           ising.get_energy(couplings, local_fields, state)))

        energy += de
        state_copy = np.copy(state)


@pytest.mark.parametrize(
    'n_vars', [n_vars for n_vars in range(32, 129, 32) for _ in range(20)])
def test_sequential_sweep_ising(n_vars: int):

    # Generate random problem
    couplings, local_fields = gen_random_problem(n_vars, dtype=dtype)

    # Get initial state
    state = 2 * np.random.randint(2, size=n_vars).astype(dtype) - 1
    state_copy = np.copy(state)

    # Get energy
    energy = ising.get_energy(couplings, local_fields, state)

    # Apply sweep (infinite temperature)
    de = simulation.sequential_sweep(ising.update_spin, couplings, local_fields,
                                     state, 0)

    # Update must happen
    assert (not np.allclose(state, state_copy))

    # Check updated energy
    assert (np.isclose(energy + de,
                       ising.get_energy(couplings, local_fields, state)))


@pytest.mark.parametrize(
    'n_vars', [n_vars for n_vars in range(32, 129, 32) for _ in range(20)])
def test_random_sweep_ising(n_vars: int):

    # Generate random problem
    couplings, local_fields = gen_random_problem(n_vars, dtype=dtype)

    # Get initial state
    state = 2 * np.random.randint(2, size=n_vars).astype(dtype) - 1
    state_copy = np.copy(state)

    # Get energy
    energy = ising.get_energy(couplings, local_fields, state)

    # Apply sweep (infinite temperature)
    de = simulation.random_sweep(ising.update_spin, couplings, local_fields,
                                 state, 0)

    # Update must happen
    assert (not np.allclose(state, state_copy))

    # Check updated energy
    assert (np.isclose(energy + de,
                       ising.get_energy(couplings, local_fields, state)))


@pytest.mark.parametrize(
    'n_vars,n_replicas',
    [(n_vars, 4) for n_vars in range(32, 129, 32) for _ in range(20)])
def test_pt(n_vars: int, n_replicas: int):

    # Generate random problem
    couplings, local_fields = gen_random_problem(n_vars, dtype=dtype)

    # Get random temperatures
    betas = np.random.random(n_replicas) * 20
    betas_copy = np.copy(betas)
    beta_idx = np.arange(n_replicas)

    # Get initial state
    states = 2 * np.random.randint(2,
                                   size=(n_replicas, n_vars)).astype(dtype) - 1
    states_copy = np.copy(states)

    # Compute energies
    energies = np.array(
        [ising.get_energy(couplings, local_fields, state) for state in states])
    energies_copy = np.copy(energies)

    # Apply PT
    utils.pt(states, energies, beta_idx, betas)

    # Check that betas are just shuffled
    assert (np.allclose(sorted(betas), sorted(betas_copy)))

    # Check that energies are just shuffled
    assert (np.allclose(sorted(energies), sorted(energies_copy)))

    # Check that states are just shuffled
    assert (sorted(
        [''.join([str(int(x)) for x in state]) for state in states]) == sorted(
            [''.join([str(int(x)) for x in state]) for state in states_copy]))

    # Check energy are preserved
    assert (np.alltrue([
        np.isclose(energies[k],
                   ising.get_energy(couplings, local_fields, states[k]))
        for k in range(n_replicas)
    ]))


@pytest.mark.parametrize(
    'n_vars', [n_vars for n_vars in range(12, 21, 2) for _ in range(1)])
def test_random_sweep_simulation_ising(n_vars: int):

    # Generate random problem
    couplings, local_fields = gen_random_problem(n_vars, dtype=dtype)

    # Find minimum energy by bruteforce
    min_energy = get_min_energy(couplings, local_fields)

    # Fix temperature
    betas = np.array([1], dtype=dtype)
    beta_idx = np.arange(1)
    # Get initial state
    states = 2 * np.random.randint(2, size=(1, n_vars)).astype(dtype) - 1

    # Compute energies
    energies = np.array(
        [ising.get_energy(couplings, local_fields, state) for state in states])

    # Simulate
    _, (best_state, best_energy, _, _) = simulation.simulation_sequential(
        ising.update_spin, simulation.random_sweep, couplings, local_fields,
        states, energies, beta_idx, betas, 10000)

    # Check that best energy is correct
    assert (np.isclose(best_energy,
                       ising.get_energy(couplings, local_fields, best_state)))

    # Best energy should be always larger than the minimum energy
    assert (np.round(best_energy, 6) >= np.round(min_energy, 6))


@pytest.mark.parametrize(
    'n_vars', [n_vars for n_vars in range(12, 21, 2) for _ in range(1)])
def test_sequential_sweep_simulation_ising(n_vars: int):

    # Generate random problem
    couplings, local_fields = gen_random_problem(n_vars, dtype=dtype)

    # Find minimum energy by bruteforce
    min_energy = get_min_energy(couplings, local_fields)

    # Fix temperature
    betas = np.array([1], dtype=dtype)
    beta_idx = np.arange(1)
    # Get initial state
    states = 2 * np.random.randint(2, size=(1, n_vars)).astype(dtype) - 1

    # Compute energies
    energies = np.array(
        [ising.get_energy(couplings, local_fields, state) for state in states])

    # Simulate
    _, (best_state, best_energy, _, _) = simulation.simulation_sequential(
        ising.update_spin, simulation.sequential_sweep, couplings, local_fields,
        states, energies, beta_idx, betas, 10000)

    # Check that best energy is correct
    assert (np.isclose(best_energy,
                       ising.get_energy(couplings, local_fields, best_state)))

    # Best energy should be always larger than the minimum energy
    assert (np.round(best_energy, 6) >= np.round(min_energy, 6))
