# PySA WalkSAT Solver

This module implements the WalkSAT algorithm for PySA, a heuristic optimizer for SAT and MAX-SAT formulas. 
SAT problems are accepted in standard CNF format.

## Getting Started (Applications)

The standalone command line application is configured using CMake.
A C++17 compliant compiler is required. To build all targets, including examples and tests,

```
$ cmake -DCMAKE_BUILD_TYPE=Release -S . -B ./build-release/ && cmake --build ./build-release
```

The executable is built to `./build-release/src/walk-sat.x `.

The Walk-SAT application is a heuristic solver. Its algorithmic parameter is
the walk probability `p`, which sets the probability of randomly flipping any
variable in an unsatisfied clause when no variable in the clause will cause
other clauses to become unsat when flipped. The algorithm is restarted if no
solution is found after `max_steps`, up until `cutoff_time` seconds have
elapsed.

```text
Usage: walk-sat.x cnf_file max_steps [p = 0.5] [unsat = 0] [seed = 0] [cutoff_time = 0]
Find solutions to a SAT formula in CNF format using WalkSAT.

   max_steps    Maximum number of steps for each restart.
   p            WalkSAT probability of a walk move. (default = 0.5)
   unsat        Maximum number of unsatisfied clauses to target (default = 0)
   seed         Random seed (default = seed from entropy)
   cutoff_time  The cutoff benchmarking time in seconds (default = exit on first solution)

```
