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

import numpy as np
import pytest
import warnings
from pysa.sa import Solver


def generate_ising_instances(n_vars, n_instances, rng=None):
    rng = np.random.default_rng(rng)
    # Generate random +-1 problems
    problems = rng.integers(0, 2, (n_instances, n_vars, n_vars))
    problems = (problems + problems.transpose((0, 2, 1))) % 2
    problems = 2 * problems - 1
    return problems


def pysa_solve(problem,
               n_sweeps=256,
               n_replicas=40,
               n_repetitions=101,
               min_temp=1.0,
               max_temp=3.5,
               float_type='float32',
               verbose=True):
    solver = Solver(problem=problem,
                    problem_type='ising',
                    float_type=float_type)
    res = solver.metropolis_update(
        num_sweeps=n_sweeps,
        num_reads=n_repetitions,
        num_replicas=n_replicas,
        update_strategy='sequential',
        min_temp=min_temp,
        max_temp=max_temp,
        initialize_strategy='ones',
        recompute_energy=False,
        sort_output_temps=True,
        parallel=False,  # Solve serially for benchmarking purposes
        verbose=verbose)
    return res


def find_ground_states(problem_instances,
                       n_sweeps=64,
                       n_replicas=40,
                       n_reps=100,
                       min_temp=1.0,
                       max_temp=3.5):
    """
    Find the ground state energies of a collection of Ising problem instances.
    :return: gs_energies: Array of ground state energies.
    """
    gs_energies = []
    for i, problem in enumerate(problem_instances):
        _res = pysa_solve(problem,
                          n_sweeps=n_sweeps,
                          n_replicas=n_replicas,
                          n_repetitions=n_reps,
                          min_temp=min_temp,
                          max_temp=max_temp,
                          verbose=False)
        best_energies = _res.best_energy
        gs_e = np.min(best_energies)
        gs_energies.append(gs_e)
        if np.sum(np.greater(gs_e, best_energies)) > 0:
            print(f"Instance {i}: ({gs_e}, {np.max(best_energies)})")

    gs_energies = np.asarray(gs_energies)
    return gs_energies


class solverbench:

    def __init__(self,
                 problem_instances,
                 gs_energies,
                 nsw=16,
                 nreplicas=32,
                 min_temp=1.0,
                 max_temp=3.5,
                 n_repetitions=201,
                 instance_set_name="IsingInstances"):
        # Gather all metrics for each instance
        instance_runtimes = []  # Runtime arrays for each repetition, in seconds
        runs_attempted = []  # Number of total attempted repetitions
        runs_solved = []  # Number of repetitions solved successfully
        energy_gap = []  # Optimization gap (Energy of the Ising Hamiltonian)
        configurations = []  # List of configurations found
        self.set_name = instance_set_name
        self.nprobs = len(problem_instances)
        for i, problem in enumerate(problem_instances):
            _res = pysa_solve(problem,
                              n_sweeps=nsw,
                              n_replicas=nreplicas,
                              n_repetitions=n_repetitions,
                              min_temp=min_temp,
                              max_temp=max_temp,
                              verbose=False)
            # We throw out the first run to ignore the influence of PySA's initialization/JIT compilation
            runs_attempted.append(n_repetitions - 1)
            best_energies = _res.best_energy[1:]
            success_arr = np.less_equal(best_energies, gs_energies[i])
            runs_solved.append(int(np.sum(success_arr)))
            configurations.append(
                [[int(si) for si in s] for s in _res.best_state[1:]])
            runtimes = list(_res['runtime (us)'][1:] * 1e-6)
            instance_runtimes.append(runtimes)
            energy_gap.append(
                [float(e) for e in np.asarray(best_energies - gs_energies[i])])

        self.instance_runtimes = instance_runtimes
        self.runs_attempted = runs_attempted
        self.runs_solved = runs_solved
        self.energy_gap = energy_gap
        self.configurations = configurations

    def serialize(self):
        """
        Serialize the attributes of this benchmark as a list of dictionaries.
        The returned data is compatible with serialization with json.dump
        :return:
        """
        _ser = []
        for i in range(self.nprobs):
            _ser.append({
                "set": self.set_name,
                "instance_idx": i,
                "runs_attempted": self.runs_attempted[i],
                "runs_solved": self.runs_solved[i],
                # "configurations": self.configurations[i], # Don't need the configurations for the demo
                "runtime_seconds": self.instance_runtimes[i],
                "energy_gap": self.energy_gap[i]
                # Difference to the true minimum of the optimization cost function
                # For Ising, this is the energy gap, while for SAT,
                # this would instead be the number of unsatisfied clauses.
            })
        return _ser


def benchmark_pysa_solver(problem_instances,
                          gs_energies,
                          nsw=16,
                          nreplicas=32,
                          min_temp=1.0,
                          max_temp=3.5,
                          n_repetitions=201):
    bench = solverbench(problem_instances,
                        gs_energies,
                        nsw=nsw,
                        nreplicas=nreplicas,
                        min_temp=min_temp,
                        max_temp=max_temp,
                        n_repetitions=n_repetitions)
    return bench.serialize()


def nts(success_arr: np.ndarray, ptgt=0.99):
    """
    Calculate the NTS (number of iterations/restarts to solution)
    for an arbitrary array of success/failures, using the success probability
    along the final axis.
    :param success_arr: Boolean array, shape [..., N]
    :param ptgt: Target confidence level (Default: 99%)
    :return: NTS array, with dimensions success_arr.shape[:-1].

    The NTS is a float array and is not integer rounded/truncated.
    If the success probability is zero, the NTS is +inf .
    If the success probability is greater than ptgt, then the NTS is 1.0 .
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        success_arr = np.asarray(success_arr)
        nsucc = np.sum(success_arr, axis=-1)
        psucc = nsucc / success_arr.shape[-1]
        _nts = np.where(np.less(psucc, ptgt),
                        np.log(1.0 - ptgt) / np.log(1.0 - psucc), 1.0)
        _nts = np.where(np.greater(psucc, 0.0), _nts, np.inf)
        return _nts


def raw_to_tts(raw_data, set_name=None, gap_key='energy_gap', ptgt=0.99):
    """
    Gather an instance set from the raw data and calculate per-instance metrics
    :param raw_data: A list[dict[...]] Python object of the raw benchmark data deserialized from JSON
    :param set_name: Instance Set name to gather. If None, every instance in the list is gathered
    :param gap_key: Key name for the optmization gap, e.g. 'energy_gap' for Ising,
        'n_unsat_clauses' for SAT, etc...
    :param ptgt: Target confidence level to calculate NTS and TTS (Default: 99%)
    :return: Processed instance data with TTS information as a dict of numpy arrays.
    """
    instance_idx = np.asarray([
        _r['instance_idx']
        for _r in raw_data
        if (set_name is None) or (_r['set'] == set_name)
    ])
    runs_attempted = np.asarray([
        _r['runs_attempted']
        for _r in raw_data
        if (set_name is None) or (_r['set'] == set_name)
    ])
    runs_solved = np.asarray([
        _r['runs_solved']
        for _r in raw_data
        if (set_name is None) or (_r['set'] == set_name)
    ])
    runtime_seconds = np.stack([
        np.asarray(_r['runtime_seconds'])
        for _r in raw_data
        if (set_name is None) or (_r['set'] == set_name)
    ],
                               axis=0)
    runtime_mean_seconds = np.mean(runtime_seconds, axis=1)
    opt_gap = np.stack([
        np.asarray(_r[gap_key])
        for _r in raw_data
        if (set_name is None) or (_r['set'] == set_name)
    ],
                       axis=0)
    success_array = np.less_equal(opt_gap, 0.0)
    for i in range(len(runs_attempted)):
        assert runs_attempted[i] == len(success_array[i])
        assert runs_solved[i] == np.sum(success_array[i])
    success_prob = np.sum(success_array, axis=1) / runs_attempted

    _nts = nts(success_array, ptgt)
    _tts = _nts * runtime_mean_seconds

    return {
        'set': set_name,
        'ptgt': ptgt,
        'instance_idx': instance_idx,
        'runs_attempted': runs_attempted,
        'runs_solved': runs_solved,
        'runtime_seconds': runtime_seconds,
        'runtime_mean_seconds': runtime_mean_seconds,
        gap_key: opt_gap,
        'success_prob': success_prob,
        'nts': _nts,
        'tts': _tts
    }


def tts_boots(bench_tts, nboots=20, ptgt=0.99):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ninst = len(bench_tts['energy_gap'])
        nrepetitions = len(bench_tts['energy_gap'][0])
        _tts_samps = []
        for i in range(nboots):  # For each bootstrap sample
            _ttsarr = []
            for j in range(ninst):  # For each instance
                reps = np.random.choice(np.arange(nrepetitions), nrepetitions)
                success_arr = np.less_equal(bench_tts['energy_gap'][j, reps], 0)
                nsucc = np.sum(success_arr, axis=-1)
                psucc = nsucc / success_arr.shape[-1]
                _nts = np.where(np.less(psucc, ptgt),
                                np.log(1.0 - ptgt) / np.log(1.0 - psucc), 1.0)
                _nts = np.where(np.greater(psucc, 0.0), _nts, np.inf)
                _ttsarr.append(bench_tts['runtime_mean_seconds'][j] * _nts)
            _tts = np.quantile(_ttsarr, ptgt)
            _tts_samps.append(_tts)

        return np.asarray(_tts_samps)


@pytest.mark.parametrize('n_vars', [48, 64, 80])
def test_tts(n_vars: int):
    # Generate the benchmark instances
    rng = np.random.default_rng(1234 + n_vars)
    problem_instances = list(generate_ising_instances(n_vars, 50, rng))
    print("Finding the ground state energies ...")
    gs_energies = find_ground_states(problem_instances)
    print("Benching ...")
    bench = benchmark_pysa_solver(problem_instances,
                                  gs_energies,
                                  32,
                                  30,
                                  1.0,
                                  3.5,
                                  n_repetitions=101)

    bench_tts = raw_to_tts(bench, gap_key='energy_gap', ptgt=0.9)
    bench_tts_samps = tts_boots(bench_tts, 50, 0.9)
    _mu, _std = np.mean(bench_tts_samps), np.std(bench_tts_samps)
    print(f"The TTS(90%) across all instances is {_mu:5.4f} +- {_std:5.4} s")
