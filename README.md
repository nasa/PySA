# PySA-Branching


## Getting Started

PySA-Branching is a header-only C++17 library for implementing branch-and-bound optimization procedures 
such as the DPLL algorithm. A minimal example implementation is provided as the target `test_branching.x`.

## Contributors

[Salvatore Mandrà](https://github.com/s-mandra)<br>
[Humberto Munoz-Bauza](https://github.com/hmunozb)<br>

## Multithreading and MPI 

**PySA-Branching** uses standard C++ concurrency to parallelize generic branch exploration.
Additionally, it supports distributing branches using `MPI` processes when compiled with `MPI` support by
passing `-DMPI=ON` to CMake, and optionally `-DMPIEXEC_EXECUTABLE=/path/to/mpiexec`.

## Licence

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
