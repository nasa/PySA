# PySA: Fast Simulated Annealing in Native Python

[![CI](https://github.com/SECQUOIA/PySA/actions/workflows/ci.yml/badge.svg)](https://github.com/SECQUOIA/PySA/actions/workflows/ci.yml)
[![Code Coverage](https://github.com/SECQUOIA/PySA/actions/workflows/coverage.yml/badge.svg)](https://github.com/SECQUOIA/PySA/actions/workflows/coverage.yml)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**PySA** is an extensible platform to optimize classical cost function.

PySA is a heuristic solver that does not provide bounds or guarantees on optimality of solutions.

## Getting Started

Tutorials on how to use **PySA** can be found in
[pysa/tutorials](https://github.com/nasa/pysa/tree/main/tutorials).

## Contributors

[Salvatore Mandrà](https://github.com/s-mandra) (PySA core)<br>
[Humberto Munoz-Bauza](https://github.com/hmunozb) (PySA core)<br>
[Ata Akbari Asanjan](https://github.com/) (Restricted Boltzmann Machine)<br>
[Lucas Brady](https://github.com/) (Restricted Boltzmann Machine)<br>
[Aaron Lott](https://github.com/) (Restricted Boltzmann Machine)<br>
[David Bernal Neira](https://github.com/bernalde) (Hyper Parameters Optimization)<br>

## Available Modules

**PySA** provides multiple modules that can be independently installed:
* Generator of Random McEliece instances ([pysa-mceliece](https://github.com/nasa/PySA/tree/pysa-mceliece))
* Stern Algorithm to solve unstructured decoding problems ([pysa-stern](https://github.com/nasa/PySA/tree/pysa-stern))
* Branch-and-Bound Library with Workload Autobalance Between Threads and Nodes ([pysa-branching](https://github.com/nasa/PySA/tree/pysa-branching))
* WalkSAT algorithm for SAT problems ([pysa-walksat](https://github.com/nasa/PySA/tree/pysa-walksat))
* DPLL Implementation for SAT based on `pysa-braching` ([pysa-dpll](https://github.com/nasa/PySA/tree/pysa-dpll))

## Installation

**PySA** can be installed by using `pip` directly from the `GitHub` 
repository:
```
pip install git+https://github.com/nasa/pysa
```
using a `zip` file:
```
pip install pysa.zip
```
or by using `conda`:
```
conda env create -f environment.yml
```

## Testing and CI

**PySA** uses GitHub Actions for continuous integration. The CI pipeline includes:

- **Automated Testing**: Unit tests run on Python 3.8 through 3.12
- **Code Coverage**: Coverage reports are generated and tracked
- **Code Formatting**: YAPF ensures consistent code style
- **Examples Validation**: All example scripts are tested automatically
- **Tutorial Execution**: Jupyter notebooks are executed to ensure they work

### Running Tests Locally

To run the test suite locally:

```bash
pip install -e . pytest pytest-cov
pytest tests/
```

To generate a coverage report:

```bash
pytest --cov=pysa --cov-report=html tests/
```

### CI Workflows

The repository includes several CI workflows:

- **CI** (`ci.yml`): Main comprehensive workflow that runs all checks
- **Code Coverage** (`coverage.yml`): Generates detailed coverage reports
- **Python PyTest** (`python-pytest.yml`): Runs unit tests across all Python versions
- **Python Code Formatting** (`python-yapf.yml`): Validates code formatting
- **Python Examples** (`python-example.yml`): Tests example scripts
- **Python Tutorials** (`python-tutorials.yml`): Executes tutorial notebooks

All workflows are triggered on pushes to `main` and on pull requests.

## NASA Open Source Agreement and Contributions

See [NOSA](https://github.com/nasa/pysa/tree/main/docs/nasa-cla/).

## Notices

Copyright @ 2023, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

## Disclaimer

_No Warranty_: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

_Waiver and Indemnity_:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST
THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS
ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE,
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT. 

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

