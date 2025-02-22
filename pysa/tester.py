import numpy as np
from sa import Solver

qubo_matrix = np.array([
    [1, -1,  2,  0],
    [-1, 2, -3,  1],
    [2, -3,  4, -2],
    [0,  1, -2,  3]
])

solver = Solver(problem=qubo_matrix,problem_type="qubo")

results = solver.metropolis_update(num_sweeps=1000,num_reads=5)
print("Final Results")
print(results)