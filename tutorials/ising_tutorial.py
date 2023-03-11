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
from pysa.sa import Solver


class _Ising2D:
    """
        Base class for 2D Ising Model.

        Parameters:
            size (list): size of the lattice
            interaction (float): strength of the nearest neighbor interaction
            magnetic_field (numpy array): external magnetic field, if None, there will be no external magnetic field
            mu (float): magnetic moment of the spins
    """

    def __init__(self, size=[], interaction=1, magnetic_field=None, mu=1):
        self.size = size
        self.interaction = interaction
        self.magnetic_field = magnetic_field
        self.mu = mu
        self._lattice = np.random.choice(np.array([-1, 1]), size=self.size)

    def __getitem__(self, *key):
        """
            Allow lattice[(i,j)] lookup
        """
        return self._lattice[key]

    def __str__(self):
        """
            String representation of the lattice for debugging purposes
        """
        return str(self._lattice)

    def nearest_neighbours(self, x, y):
        """
            Calculate the sum of the nearest neighbours' spin
            
            Parameters:
                x(int): x-coordinate of the spin
                y(int): y-coordinate of the spin
            
            Returns:
                float: sum of the nearest neighbours' spin
        """
        return self._lattice[(x - 1) % self.size[0], y] + \
            self._lattice[(x + 1) % self.size[0], y] + \
            self._lattice[x, (y - 1) % self.size[1]] + \
            self._lattice[x, (y + 1) % self.size[1]]

    def flip_spin(self, i, j):
        """
            Flip the spin of the lattice at (i, j)
            
            Parameters:
                i(int): x-coordinate of the spin
                j(int): y-coordinate of the spin
        """
        self._lattice[i, j] *= -1

    def hamiltonian_energy(self, x, y):
        """
            Calculate the Hamiltonian energy of the spin at (x, y)
            
            Parameters:
                x(int): x-coordinate of the spin
                y(int): y-coordinate of the spin
            
            Returns:
                float: Hamiltonian energy of the spin at (x, y)
        """
        if self.magnetic_field is not None:
            return -self.interaction * self._lattice[
                x, y] * self.nearest_neighbours(x, y) - self.mu * self._lattice[
                    x, y] * self.magnetic_field[x, y]
        else:
            return -self.interaction * self._lattice[
                x, y] * self.nearest_neighbours(x, y)


class Ising2DwithMetroplis(_Ising2D):

    def __init__(self, size=[], interaction=1, magnetic_field=None, mu=1):
        """
            Initialize the Ising2DwithMetroplis object.
            
            Parameters:
                size (list): the size of the lattice, which is the number of rows and columns.
                interaction (int): the number of nearest neighbors.
                magnetic_field (ndarray): an external magnetic field, represented as an array of the same size as the lattice.
                mu (int): a coefficient for the external magnetic field.
        """
        super().__init__(size, interaction, magnetic_field, mu)

    def update(self, temperature, sweeps=1000, verbose=False, intervals=1000):
        """
            Update the Ising lattice via the Markov Chain Monte Carlo (MCMC).
            
            Parameters:
                temperature (float): the temperature of the system.
                sweeps (int): the number of MCMC sweeps.
                verbose (bool): if True, the function will print the sweep number, the chosen lattice point, 
                and the result of the spin flip.
                intervals (int): the number of intervals for saving the history of the lattice.
            
            Returns:
                hist (ndarray): the history of the lattice, which is an array of shape (sweeps//intervals, size[0], size[1]).
        """
        hist = np.zeros([sweeps // intervals] + self.size)
        # Calculate beta which is inverse temperature
        beta = 1 / temperature

        # Create the chains of MCMC
        cnt = 0
        for sweep in range(sweeps):
            i = np.random.randint(
                self.size[0])  # Randomly selecting an individual lattice point
            j = np.random.randint(self.size[1])
            dH = self.hamiltonian_energy(i, j)  # Calculate the energy of the

            if (dH > 0):
                boltzmann = np.exp(-1 * beta * dH)
                randNum = np.random.uniform()
                result = True
                if boltzmann <= randNum:
                    self.flip_spin(
                        i, j)  # you where not calling the method properly
                    result = False
            else:
                boltzmann = 1
                randNum = 1
                result = True

            # some console output
            if verbose:
                print(sweep, i, j, result)
                print("dh: ", dH)

            if sweep % intervals == 0:
                hist[cnt] = self._lattice.copy()
                cnt += 1
        return hist


class Ising2DwithPySA(_Ising2D):

    def __init__(self, size=[], interaction=1, magnetic_field=None, mu=1):
        """
            Initialize the Ising2DwithPySA object.
            
            Parameters:
                size (list): the size of the lattice, which is the number of rows and columns.
                interaction (int): the number of nearest neighbors.
                magnetic_field (ndarray): an external magnetic field, represented as an array of the same size as the lattice.
                mu (int): a coefficient for the external magnetic field.
        """
        super().__init__(size, interaction, magnetic_field, mu)

    @staticmethod
    def matrix_to_adjacency(matrix):
        m, n = matrix.shape
        adjacency = np.zeros((m * n, m * n))
        for i in range(m):
            for j in range(n):
                if i > 0:
                    adjacency[i * n + j, (i - 1) * n + j] = 1
                if i < m - 1:
                    adjacency[i * n + j, (i + 1) * n + j] = 1
                if j > 0:
                    adjacency[i * n + j, i * n + j - 1] = 1
                if j < n - 1:
                    adjacency[i * n + j, i * n + j + 1] = 1
        return adjacency

    def update(self,
               sweeps=1000,
               reads=100,
               verbose=False,
               temp_range=[1.0, 3.5],
               replicas=40):
        """
            Update the Ising lattice via the Python Simulated Annealing (PySA).
            
            Parameters:
                temperature (float): the temperature of the system.
                sweeps (int): the number of MCMC sweeps.
                verbose (bool): if True, the function will print the sweep number, the chosen lattice point, 
                and the result of the spin flip.
                intervals (int): the number of intervals for saving the history of the lattice.
            
            Returns:
                hist (ndarray): the history of the lattice, which is an array of shape (sweeps//intervals, size[0], size[1]).
        """

        # Define the Problem matrix
        if self.magnetic_field is not None:
            self.problem = self.matrix_to_adjacency(
                self._lattice) * -self.interaction
            np.fill_diagonal(self.problem,
                             self.magnetic_field.flatten() * -self.mu)
        else:
            self.problem = self.matrix_to_adjacency(
                self._lattice) * -self.interaction

        # Instantiate Pysa Solver
        self.solver = Solver(problem=self.problem,
                             problem_type='ising',
                             float_type="float32")

        # Apply Metropolis
        res = self.solver.metropolis_update(
            num_sweeps=sweeps,
            num_reads=reads,
            num_replicas=replicas,
            update_strategy='sequential',
            min_temp=temp_range[0],
            max_temp=temp_range[1],
            initialize_strategy='ones',
            recompute_energy=False,
            sort_output_temps=True,
            parallel=True,  # True by default
            verbose=verbose)

        return [
            res['states'].iloc[i][-1].reshape(self._lattice.shape)
            for i in range(len(res))
        ]


"""
##############################################################################################################
# How to test Ising model via simple MCMC

import sys
sys.path.append("../pysa/")
from ising_003 import *
import matplotlib.pyplot as plt

# Without external magnetic forces
size = [16, 16]
ising = Ising2DwithMetroplis(size=size, interaction=1, magnetic_field=None, mu=1)
_ising = ising._lattice.copy()
hist = ising.update(temperature=10, sweeps=10000, verbose=False, intervals=1000)

plt.figure(figsize=(20, 2))
for i in range(len(hist)):
    plt.subplot(1, len(hist), i+1)
    plt.imshow(hist[i], interpolation='none', cmap='gray')
    plt.axis("off")

plt.show()



# # # # With external magnetic forces
size= [16, 16]
external_force = np.ones(size) * -1
external_force[size[0]//4:(3*size[0]//4), size[1]//4:(3*size[1]//4)] = 1

ising = Ising2DwithMetroplis(size=size, interaction=1, magnetic_field=external_force, mu=0.75)
hist = ising.update(temperature=10, sweeps=40000, verbose=False, intervals=4000)

plt.figure(figsize=(20, 4))
plt.subplot(2, len(hist), 1)
plt.imshow(external_force, interpolation='none', cmap='gray')
plt.title("External\nMagnetic Force")
plt.axis("off")

for i in range(len(hist)):
    plt.subplot(2, len(hist), i+1+len(hist))
    plt.imshow(hist[i], interpolation='none', cmap='gray')
    plt.axis("off")

plt.show()


##############################################################################################################
# How to test Ising model via simple MCMC

import sys
sys.path.append("./pysa/")
from ising_003 import *
import matplotlib.pyplot as plt

size = [16, 16]
external_force = None
ising = Ising2DwithPySA(size=size, interaction=-1, magnetic_field=external_force, mu=0.75)
ising._lattice = _ising
hist = ising.update(temp_range=[1.0, 3.5], sweeps=400, reads=100, replicas=10, verbose=False)

for i in range(len(hist) // 10):
    plt.subplot(2, len(hist)//10, i+1+len(hist)//10)
    plt.imshow(hist[i], interpolation='none', cmap='gray')
    plt.axis("off")

plt.show()



# # # # With external magnetic forces
size = [16, 16]
external_force = np.ones(size) * -1
external_force[size[0]//4:(3*size[0]//4), size[1]//4:(3*size[1]//4)] = 1
ising = Ising2DwithPySA(size=size, interaction=-1, magnetic_field=external_force, mu=-0.75)
hist = ising.update(temp_range=[1.0, 3.5], sweeps=4000, reads=100, replicas=10, verbose=False)

plt.figure(figsize=(20, 4))
plt.subplot(2, len(hist)//10, 1)
plt.imshow(external_force, interpolation='none', cmap='gray')
plt.title("External\nMagnetic Force")
plt.axis("off")

for i in range(len(hist) // 10):
    plt.subplot(2, len(hist)//10, i+1+len(hist)//10)
    plt.imshow(hist[i], interpolation='none', cmap='gray')
    plt.axis("off")

plt.show()


plt.imshow(ising.problem)
plt.colorbar()
plt.grid()
plt.show()



"""
