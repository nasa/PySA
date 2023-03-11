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

from time import time
from pysa.sa import *
import pysa.ais


def generate_rnd_prob(n):
    '''Generates a random quadratic model connectivity and weight matrix.
    This just generates a symmetric nXn matrix where every entry is randomly
    chosen between -1 and 1'''

    result = 2 * np.random.random((n, n)) - 1
    result = (result + result.T) / 2

    return result


def print_prob(prob):
    '''Prints the problem (i.e. matrix) in a way that can be natively
    copied into Mathematica.'''
    n = len(prob)

    output = "{"
    for i in range(n):
        output += "{"
        for j in range(n):
            output += str(prob[i][j])
            if j < n - 1:
                output += ","
        output += "}"
        if i < n - 1:
            output += ","
    output += "}"
    print(output)


def main():
    '''Carries out the main functionality of this example
    This example, implements Annealed Importance Sampling for 
    a classical Ising model and calculates both the average energy of
    that model at a given temperature and an estimate of the partition
    function of that model.'''

    # Problem Parameters
    float_type = 'float32'
    n = 256  # number of bits in the Ising model
    prob_type = "ising"  # whether we want this to be an ising or qubo
    final_beta = 10.0  # Final inverse temperature we want to reach
    # Parameters for annealed_importance
    num_temps = 100  # number of annealing steps
    ref_prob = "uniform"  # The reference probability for start of the anneal
    num_samps = 100  # number of samples to take with AIS
    ref_args = [0.3]  # used for the bernoulli distribution, likelihood of 0/-1
    # Parameters specifically for metrpolis_step
    n_sweeps = 32
    n_replicas = num_temps
    n_reads = num_samps
    min_temp = 1 / final_beta
    max_temp = 5.0

    # Generate the problem
    problem = generate_rnd_prob(n)

    # Get solver
    solver = Solver(problem, prob_type, float_type)

    logZ0 = pysa.ais.uniform_partition_fun(n)
    print("\nOriginal Partition Function, logZ0 = " + str(logZ0) + "\n")
    '''There are three implemented ways to get the log of the final partition 
    function.  The first way is probably more accurate and relies on a stand
    alone annealed importance sampler.  This is custom built to provide good 
    estimates of the partition function and will also output samples from the 
    target distribution as well as the weights of those samples.  This does not 
    use parallel tempering.  Note that this function also supports starting distributions other than the uniform distribution.'''

    start = time()
    logZf, samples, weights = solver.annealed_importance(
        num_temps,
        num_samps,
        reference_prob=ref_prob,
        reference_args=ref_args,
        beta=final_beta)
    print("AIS Time: " + str(time() - start))

    energies = [solver.get_energy(samp) for samp in samples]

    print("AIS Partition Function, logZf = " + str(logZf))
    print("AIS Energy Expectation Value = " + str(np.dot(weights, energies)) +
          "\n")
    '''The second option is to just do a normal PySA importance sampling based 
    on parallel tempering or regular simulated annealing.  Doing this, you use 
    PySA's metropolis_update exactly as normal.  The only change is that if you 
    are using min_temp and max_temp, you need to include the new option 
    beta0=True, and if you are using a predefined temps list, you need to 
    include np.inf in your temps list.  I also highly recommend having a decent 
    number of num_reads since this determines the number of samples of the log 
    partition function that we are averaging, and too few can be problematic.'''

    start = time()
    solution = solver.metropolis_update(num_sweeps=n_sweeps,
                                        num_reads=n_reads,
                                        num_replicas=n_replicas,
                                        update_strategy='sequential',
                                        min_temp=min_temp,
                                        max_temp=max_temp,
                                        initialize_strategy='random',
                                        beta0=True,
                                        sort_output_temps=True)

    print("PT Time: " + str(time() - start))
    logZf = pysa.ais.partition_function_post(solution)

    energies = [samp[0] for samp in solution["energies"]]

    print("PT Partition Function, logZf = " + str(logZf))
    print("PT Energy Expectation Value = " + str(np.mean(energies)) + "\n")
    '''The last way is probably the simplest.  Instead of any of the above,
    just call metropolis_update with the flag get_part_fun = True.
    This will cause the output pandas DataFrame to contain a new label
    called "log_Zf" which is the estimate of the log of the partition
    function for that run.  It is further possible to combine all these
    together by passing the DataFrame to pysa.ais.combine_logZf which
    outputs a single number.
    Note that this flag takes care of the beta0 flag on its own.'''

    start = time()
    solution = solver.metropolis_update(num_sweeps=n_sweeps,
                                        num_reads=n_reads,
                                        num_replicas=n_replicas,
                                        update_strategy='sequential',
                                        min_temp=min_temp,
                                        max_temp=max_temp,
                                        initialize_strategy='random',
                                        get_part_fun=True,
                                        sort_output_temps=True)

    print("PT2 Time: " + str(time() - start))
    logZf = pysa.ais.combine_logZf(solution)

    energies = [samp[0] for samp in solution["energies"]]

    print("PT2 Partition Function, logZf = " + str(logZf))
    print("PT2 Energy Expectation Value = " + str(np.mean(energies)) + "\n")


if __name__ == "__main__":
    print(__doc__)
    main()
