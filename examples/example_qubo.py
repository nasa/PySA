#!/usr/bin/env python3
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

import os

# Set environment variables
num_threads = 4
os.environ["NUMBA_NUM_THREADS"] = str(num_threads)
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
os.environ["NUMBA_LOOP_VECTORIZE"] = "1"
os.environ["NUMBA_ENABLE_AVX"] = "1"
os.environ["NUMBA_DEBUG_CACHE"] = "1"

# In general, numba should be able to identify the right arch and features
#os.environ["NUMBA_CPU_FEATURES"] = "+sse,+sse2,+avx,+axv2"
#os.environ["NUMBA_CPU_NAME"] = "skylake"

import termplotlib as tpl
from tqdm.auto import tqdm
from time import time
from pysa.sa import Solver
import pandas as pd
import numpy as np


# Print results
def print_results(res: pd.DataFrame):
    print(res.sort_values(by='best_energy'))
    print(
        f'# Average runtime (ms): {np.mean(res["runtime (us)"])/1000:1.5f} +/- {np.std(res["runtime (us)"])/1000:1.5f}'
    )
    print(
        f'# Average number of sweeps: {np.mean(res["min_sweeps"]):1.5f} +/- {np.std(res["min_sweeps"]):1.5f}'
    )
    print("\n# Energy distribution:")
    _counts, _bins = np.histogram(res['best_energy'], bins=20)
    _fig = tpl.figure()
    _fig.hist(_counts, _bins, force_ascii=False, orientation='horizontal')
    _fig.show()
    print("\n# Runtime distribution (ms):")
    _counts, _bins = np.histogram(res['runtime (us)'] / 1000, bins=20)
    _fig = tpl.figure()
    _fig.hist(_counts, _bins, force_ascii=False, orientation='horizontal')
    _fig.show()
    print("\n# Number of sweeps distribution:")
    _counts, _bins = np.histogram(res['num_sweeps'], bins=10)
    _fig = tpl.figure()
    _fig.hist(_counts, _bins, force_ascii=False, orientation='horizontal')
    _fig.show()
    print("\n# Number of sweeps to best state distribution:")
    _counts, _bins = np.histogram(res['min_sweeps'], bins=10)
    _fig = tpl.figure()
    _fig.hist(_counts, _bins, force_ascii=False, orientation='horizontal')
    _fig.show()


def __run__():
    # Number of variables
    n_sweeps = 32
    n_vars = 256
    n_replicas = 40
    n_reads = 100
    min_temp = 0.3
    max_temp = 1.5

    # Using 'float64' is about ~20% slower
    float_type = 'float32'

    # Generate random problem
    problem = 2 * np.random.random((n_vars, n_vars)) - 1
    problem = (problem + problem.T) / 2

    # Get solver
    solver = Solver(problem=problem, problem_type='qubo', float_type=float_type)
    """
    Simulation by using fixed initialization and sequential update. This should be faster
    but it could miss the best solution because always starting from the same initial state.
    """
    print(str.center("=" * 10 + " SIM1 " + "=" * 10, 30))
    res_1 = solver.metropolis_update(num_sweeps=n_sweeps,
                                     num_reads=n_reads,
                                     num_replicas=n_replicas,
                                     update_strategy='sequential',
                                     min_temp=min_temp,
                                     max_temp=max_temp,
                                     initialize_strategy='zeros',
                                     verbose=True)
    print_results(res_1)
    """
    Simulation by using random initialization and random update. This should be slower
    but it could help to find states at lower energy by starting from random initial states.
    """
    print(str.center("=" * 10 + " SIM2 " + "=" * 10, 30))
    res_2 = solver.metropolis_update(num_sweeps=n_sweeps,
                                     num_reads=n_reads,
                                     num_replicas=n_replicas,
                                     update_strategy='random',
                                     min_temp=min_temp,
                                     max_temp=max_temp,
                                     initialize_strategy='random',
                                     verbose=True)
    print_results(res_2)
    """
    Simulation by using given initial states and random update.
    """
    print(str.center("=" * 10 + " SIM3 " + "=" * 10, 30))
    res_3 = solver.metropolis_update(num_sweeps=n_sweeps,
                                     num_reads=n_reads,
                                     num_replicas=n_replicas,
                                     update_strategy='random',
                                     min_temp=min_temp,
                                     max_temp=max_temp,
                                     initialize_strategy=np.random.randint(
                                         2, size=(n_replicas,
                                                  n_vars)).astype(float_type),
                                     verbose=True)
    print_results(res_3)


if __name__ == '__main__':
    print(__doc__)
    __run__()
