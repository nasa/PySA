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

from concurrent.futures import ThreadPoolExecutor
from typing import List, NoReturn, Tuple, Any
from tqdm.auto import tqdm
from time import time
import numpy as np
import scipy
import pandas as pd
import numba

import pysa.qubo
import pysa.ising
import pysa.simulation
import pysa.ais

Vector = List[float]
Matrix = List[List[float]]
State = List[float]


class Solver(object):

    def __init__(self,
                 problem: Matrix,
                 problem_type: str,
                 float_type: Any = 'float32') -> NoReturn:

        # Only qubo is supported
        if problem_type == 'qubo':
            self._module = pysa.qubo
        elif problem_type == 'ising':
            self._module = pysa.ising
        else:
            raise ValueError(f"problem_type='{problem_type}' not supported.")

        # Set problem_type
        self.problem_type = problem_type

        # Set number of variables and replicas
        self.n_vars = len(problem)

        # Set float type
        self.float_type = float_type

        # Check that problem is a square matrix
        if problem.shape != (self.n_vars, self.n_vars):
            raise ValueError("Problem must be a square matrix.")

        # Check that problem is symmetric
        if not np.allclose(problem, problem.T):
            raise ValueError("Problem must be a symmetric matrix.")

        # Split the problem in couplings and local_fields
        self.local_fields = np.copy(np.diag(problem))
        self.couplings = np.copy(problem)
        np.fill_diagonal(self.couplings, 0)

    def get_energy(self, conf: Vector, dtype: Any = None):
        """
        Return the energy of a given configuration.
        """

        # Set type
        if dtype is None:
            dtype = self.couplings.dtype

        return self._module.get_energy(self.couplings.astype(dtype),
                                       self.local_fields.astype(dtype),
                                       conf.astype(dtype))

    def metropolis_update_async(self, **kwargs):

        # Set send_background to False
        kwargs['send_background'] = False

        return self.metropolis_update(**kwargs)

    def metropolis_update(self,
                          num_sweeps: int,
                          num_reads: int = 1,
                          num_replicas: int = None,
                          temps: np.ndarray = None,
                          min_temp: float = 0.3,
                          max_temp: float = 1.5,
                          update_strategy: str = 'random',
                          initialize_strategy: str = 'random',
                          init_energies: List[float] = None,
                          recompute_energy: bool = False,
                          sort_output_temps: bool = False,
                          return_dataframe: bool = True,
                          parallel: bool = True,
                          use_pt: bool = True,
                          send_background: bool = False,
                          verbose: bool = False,
                          get_part_fun: bool = False,
                          beta0: bool = False) -> pd.DataFrame:
        '''This function runs a full simulated annealing importance sampling with a focus on parallel tempering.  This is the main user interface with PySA.  If return_dataframe=True, then the output will be a pandas DataFrame.  This is an object that contains a lot of data indexed by string labels, to access any of the below, you can take the output object and look for the below label index, e.g. output['temps']
                'states' = a num_reads X num_replicas array of output states 
                            each of which is length num_vars
                'energies' = a num_reads X num_replicas array of energies of the 
                            'states' listed above
                'best_state' = a num_reads array containing the lowest energy 
                            state obtained in each read
                'best_energy' = a num_reads array containing the lowest energies
                            obtained in each read (corresponding to best_state)
                'temps' = a num_reads X num_replicas array of the temperatures 
                            of each state from 'states'.  Note that this might 
                            be in a weird order, but the order is the same 
                            ordering as 'states' and 'energies'
                'num_sweeps' = a num_reads array of the number of sweeps for the 
                            corresponding read
                'min_sweeps' = a num_reads array of the minimum number of sweeps 
                            needed to obtain the best energy
                'init_time (us)' = a num_reads array of how long initialization 
                            took for each read
                'runtime (us)' = a num_reads array of how long simulated     
                            annealing took for each read
                'problem_type' = a num_reads array of the problem type used
                'float_type' = a num_reads array of the float type used
                'log_Zf' = optional and only occurs if get_part_fun = True
                            a num_reads array containing estimates of the log
                            of the partition function for each read
        '''

        # Send process in background
        if send_background:
            return ThreadPoolExecutor(max_workers=1).submit(
                Solver.metropolis_update_async, **locals())

        # Get/check number of replicas
        if not temps is None:

            if num_replicas and len(temps) != num_replicas:
                raise ValueError(f"len('temps') != 'num_replicas'.")
            else:
                num_replicas = len(temps)

        if type(initialize_strategy) != str:

            if num_replicas and len(initialize_strategy) != num_replicas:
                raise ValueError(
                    f"len('initialize_strategy') != 'num_replicas'.")
            else:
                num_replicas = len(initialize_strategy)

            if not init_energies is None and len(init_energies) != num_replicas:
                raise ValueError(f"len('init_energies') != 'num_replicas'.")

        # The partition function needs an infinite temperature reference
        if get_part_fun:
            beta0 = True
        # Set temperatures
        if temps is None:

            # If num_replicas is not set, set it to 4
            if not num_replicas:
                num_replicas = 4

            if num_replicas == 1:
                betas = np.array([1 / min_temp], dtype=self.float_type)
            elif num_replicas == 2:
                betas = np.array([1 / min_temp, 1 / max_temp],
                                 dtype=self.float_type)
            elif num_replicas > 2:
                if beta0:
                    _ratio = np.exp(1 / (num_replicas - 2) *
                                    np.log(max_temp / min_temp))
                    betas = np.array([
                        1 / (min_temp * _ratio**k)
                        for k in range(num_replicas - 1)
                    ],
                                     dtype=self.float_type)
                    betas = np.append(betas, 0.0)
                else:
                    _ratio = np.exp(1 / (num_replicas - 1) *
                                    np.log(max_temp / min_temp))
                    betas = np.array([
                        1 / (min_temp * _ratio**k) for k in range(num_replicas)
                    ],
                                     dtype=self.float_type)

        else:
            if beta0:
                if np.inf not in temps:
                    raise ValueError("get_part_fun = True or beta0 = True \
                    require the temps array to include np.inf")
            betas = 1 / np.array(temps, dtype=self.float_type)

        # Cast couplings and local_fields to the desired float
        couplings = self.couplings.astype(self.float_type)
        local_fields = self.local_fields.astype(self.float_type)

        # Select if sequential of parallel simulation
        if parallel:
            simulation = pysa.simulation.simulation_parallel
        else:
            simulation = pysa.simulation.simulation_sequential

        # Get initial states (random state)
        if type(initialize_strategy) == str:

            if initialize_strategy == 'random':

                if self.problem_type == 'qubo':

                    def _init_strategy():

                        # Initialize to random state
                        states = np.random.randint(2,
                                                   size=(num_replicas,
                                                         self.n_vars)).astype(
                                                             self.float_type)

                        # Get initial energies
                        energies = np.array([
                            self.get_energy(state, dtype=self.float_type)
                            for state in states
                        ])

                        return states, energies

                elif self.problem_type == 'ising':

                    def _init_strategy():

                        # Initialize to random state
                        states = 2 * np.random.randint(
                            2, size=(num_replicas, self.n_vars)).astype(
                                self.float_type) - 1

                        # Get initial energies
                        energies = np.array([
                            self.get_energy(state, dtype=self.float_type)
                            for state in states
                        ])

                        return states, energies

                else:

                    raise ValueError(
                        f"self.problem_type=='{self.problem_type}' not supported."
                    )

            elif initialize_strategy == 'zeros' and self.problem_type == 'qubo':

                def _init_strategy():

                    # Initialize state to zeros
                    states = np.zeros((num_replicas, self.n_vars),
                                      dtype=self.float_type)

                    # Get initial energies
                    energies = np.zeros(num_replicas, dtype=self.float_type)

                    return states, energies

            elif initialize_strategy == 'ones':

                def _init_strategy():

                    # Initialize state to ones
                    states = np.ones((num_replicas, self.n_vars),
                                     dtype=self.float_type)

                    # Get initial energies
                    energies = np.ones(num_replicas, dtype=self.float_type) * (
                        np.sum(couplings) / 2 + np.sum(local_fields))

                    return states, energies

            else:
                raise ValueError(
                    f"initialize_strategy='{initialize_strategy}' not recognized."
                )

        else:

            try:

                # Compute initial energies if not provided by the user
                if init_energies is None:

                    init_energies = np.array([
                        self.get_energy(state, dtype=self.float_type)
                        for state in initialize_strategy
                    ])

                def _init_strategy():

                    states = np.copy(initialize_strategy).astype(
                        self.float_type)
                    energies = np.copy(init_energies).astype(self.float_type)

                    return states, energies

            except:

                raise ValueError("Cannot initialize system.")

        if update_strategy == 'random':

            def _simulate_core():

                # Start timer
                t_ini = time()

                # Get states and energy of replicas
                w = _init_strategy()
                # Assign initial temperatures sequentially to the replicas
                beta_idx = np.arange(num_replicas)
                # End timer
                t_end = time()

                return simulation(self._module.update_spin,
                                  pysa.simulation.random_sweep, couplings,
                                  local_fields, *w, beta_idx, betas, num_sweeps,
                                  get_part_fun, use_pt), t_end - t_ini

        elif update_strategy == 'sequential':

            def _simulate_core():

                # Start timer
                t_ini = time()

                # Get states and energy
                w = _init_strategy()
                # Assign initial temperatures sequentially to the replicas
                beta_idx = np.arange(num_replicas)

                # End timer
                t_end = time()

                return simulation(self._module.update_spin,
                                  pysa.simulation.sequential_sweep, couplings,
                                  local_fields, *w, beta_idx, betas, num_sweeps,
                                  get_part_fun, use_pt), t_end - t_ini

        else:
            raise ValueError(
                f"update_strategy='{update_strategy}' not recognized.")

        def _simulate():

            # Get initial time
            t_ini = time()

            # Simulate
            ((out_states, out_energies, out_beta_idx, out_log_omegas),
             (best_state, best_energy, best_sweeps,
              ns)), init_time = _simulate_core()

            # Get final time
            t_end = time()

            # Recompute energies to the original precision (if required)
            if recompute_energy:

                best_energy = self.get_energy(best_state)

                out_energies = np.array(
                    [self.get_energy(state) for state in out_states])

            # Sort output temperatures (if required)
            if sort_output_temps:

                # Sort temperatures and states
                out_betas = betas
                out_states = out_states[out_beta_idx]
                out_energies = out_energies[out_beta_idx]
            else:
                out_betas = betas[out_beta_idx]

            with np.errstate(divide='ignore'):
                out_temps = np.divide(1, out_betas)
            # Energy is recomputed without recasting to the specified float_type
            if get_part_fun:
                logZf = pysa.ais.omegas_to_partition(out_log_omegas,
                                                     self.n_vars * np.log(2))
                return {
                    'states': out_states,
                    'energies': out_energies,
                    'best_state': best_state,
                    'best_energy': best_energy,
                    'temps': out_temps,
                    'log_Zf': logZf,
                    'num_sweeps': ns,
                    'min_sweeps': best_sweeps,
                    'init_time (us)': int(init_time * 1e6),
                    'runtime (us)': int((t_end - t_ini - init_time) * 1e6),
                    'problem_type': self.problem_type,
                    'float_type': self.float_type
                }
            else:
                return {
                    'states': out_states,
                    'energies': out_energies,
                    'best_state': best_state,
                    'best_energy': best_energy,
                    'temps': out_temps,
                    'num_sweeps': ns,
                    'min_sweeps': best_sweeps,
                    'init_time (us)': int(init_time * 1e6),
                    'runtime (us)': int((t_end - t_ini - init_time) * 1e6),
                    'problem_type': self.problem_type,
                    'float_type': self.float_type
                }

        # Run simulations
        _res = [
            _simulate() for _ in tqdm(np.arange(num_reads), disable=not verbose)
        ]

        # Return results
        return pd.DataFrame(_res) if return_dataframe else _res

    def annealed_importance(self,
                            num_temps: int,
                            num_samps: int,
                            reference_prob: "str",
                            update_strategy: str = 'random',
                            reference_args=None,
                            beta: float = 1.0,
                            parallel: bool = True):
        '''This goes through and does the full annealed importance sampling
        INPUTS
        num_temps = the number of temperature steps in the anneal
        num_samps = the number of annealing samples taken
        reference_prob = a string representing which reference probability
                        you want to be the starting point of the anneal.
                        See pysa.ais for implementations, currently only "uniform" or "bernoulli"
        update_strategy = string dictating how the metropolis steps are taken.
                        For each temperature, we sweep through the entire state
                        proposing a change with every bit, either "sequential"
                        or "random"
        reference_args = an optional input that can be used to give additional
                        info about the reference probability distribution.
                        For "uniform", this does nothing
                        For "bernoulli" this should be a list of one float 
                        number that is the probability of 0 or -1 depending on 
                        the model being used
        beta = The inverse temperature that we want the final distribution to be
                        at.  Note that this is independent of num_temps or the
                        annealing behavior.
        parallel = a Boolean that determines if we parallelize this
        OUTPUTS
        logZf = an estimate of the log of the partition function of the final 
                probability distribution
        samples = The actual samples taken using annealed importance sampling
        weights = The weights associated with each of the samples above
        '''

        # Set some things up based on the reference probability
        if reference_prob == "uniform":
            # Initial Partition Fun
            logZ0 = pysa.ais.uniform_partition_fun(self.n_vars)

            # State Initialization
            initialize_prob = pysa.ais.uniform_prob_initialization
            initial_args = None

            # The reference probability distribution
            ref_prob = pysa.ais.uniform_ref_prob
        elif reference_prob == "bernoulli":
            # Initial Partition Fun
            logZ0 = pysa.ais.bernoulli_partition_fun(self.n_vars)

            # State Initialization
            initialize_prob = pysa.ais.bernoulli_prob_initialization
            initial_args = reference_args

            # The reference Probability
            bernoulli_p = initial_args[0]

            @numba.njit(fastmath=True, nogil=True, parallel=False)
            def ref_prob(state: Vector, pos: int = -1):
                return pysa.ais.bernoulli_ref_prob(state, bernoulli_p, pos)
        else:
            raise ValueError("Reference Probability not recognized.")

        # determine the update strategy in the sweep
        if update_strategy == 'random':
            sweep_func = pysa.simulation.random_sweep_ref
        elif update_strategy == 'sequential':
            sweep_func = pysa.simulation.sequential_sweep_ref
        else:
            raise ValueError(
                f"update_strategy='{update_strategy}' not recognized.")

        # create a list of temperatures to anneal through
        temps = [(i + 1) / (num_temps) for i in range(num_temps)]
        temps = np.array(temps, dtype=self.float_type)

        # Set the types of couplings and local fields correctly
        # This also scales them based on the final desired temperature
        couplings = beta * self.couplings.astype(self.float_type)
        local_fields = beta * self.local_fields.astype(self.float_type)

        # Set the spin update rule
        update_spin = self._module.update_spin_ref
        # Set the energy function
        if self.problem_type == 'qubo':
            energy_fun = pysa.qubo.get_energy
        elif self.problem_type == 'ising':
            energy_fun = pysa.ising.get_energy
        else:
            raise ValueError(f"'{self.problem_type}' not supported for AIS.")

        # Choose the simulation function
        if parallel:
            simulate = pysa.simulation.ais_simulation_parallel
        else:
            simulate = pysa.simulation.ais_simulation_sequential

        initial_states = []
        for i in range(num_samps):
            initial_states += [
                initialize_prob(self.n_vars, self.problem_type,
                                initial_args).astype(self.float_type)
            ]
        initial_states = np.array(initial_states)

        samples, log_omegas = simulate(temps, update_spin, sweep_func, ref_prob,
                                       energy_fun, couplings, local_fields,
                                       initial_states)

        # Get the partition function
        logZf = pysa.ais.omegas_to_partition(np.array(log_omegas), logZ0)

        # normalize the weights we found
        normalization = 0  # the factor to normalize the omegas
        lomax = log_omegas.max()
        for log_o in log_omegas:
            normalization += np.exp(log_o - lomax)
        omegas = [np.exp(log_omegas[i]-lomax)/normalization\
                    for i in range(num_samps)]

        return logZf, samples, omegas
