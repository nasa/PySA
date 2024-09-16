Stern Algorithm Solver
======================

Installation
----

`sternx` runs the Stern algorithm to solve unstructured decoding and *p*-spin
Hamiltonian ground states.

Requirements: C++17 and Fortran compilers (such as gcc and gfortran) and Boost.
Optionally, the program can be built with an MPI compiler.

```sh
mkdir build && cd build
cmake .. 
make sternx
```

To compile with MPI support, run cmake with the definition `cmake -DMPI ..`.

Input file format
----

The MLD file format describes a maximum likelihood decoding problem using a
CNF-like specification, where each line describes a row of the linear equation
over $\mathbb{F}_2$. The overall format is summarized as follows:

```txt
(g|h) N M y w
 1 2 3 4
-2 3 4 5
...
```

The first line is a header with the following data:

- A single character, `g` or `h`
      - If `g`, the file describes the $n\times k$ generator matrix transpose
        for a $[n, k]$ code with `M = n` clauses and `N = k` variables.
      - If `h`, the file describes the parity check matrix for a $[n, k]$ code
        with `M = n-k` clauses and `N = n` variables.
      - The characters `p` and `w` are reserved.
- `N` Number of column variables.
- `M` Number of rows, i.e. clauses.
- `y = (0|1)` Sets the boolean assignment for each row to either false (`0`) or
  true (`1`).
- `w` Hamming weight of the code space error.
      - `w > 0` Hamming weight is exactly `w`
      - `w == 0` Hamming weight is not specified
      - `w < 0` Hamming weight is at most `|w|`

Lines 2 through `M+1` specify the non-zero elements of the matrix using 1-based
indices (1 to `N`). If an integer is negative, then $y_i$ is negated to $(1 -
y_i)$.

As an example, the Niederreiter instances contain the header `h n M 1 t`, where
`M=n-k` and `t` is the error correction distance of the Goppa decoder. Negated
lines correspond to rows where $y_i = 0$, and other lines correspond to rows
where $y_i = 1$.

*p*-spin Hamiltonian Ground State
----

A *p*-spin Hamiltonian over $k$ variables has the general form

$$ H = -\sum_{i=1}^N J_i \sigma_{i_1} \ldots \sigma_{i_p}. $$

When $J_i=\pm 1$, the problem of finding the ground state of this Hamiltonian
can be reduced to a maximum likelihood decoding problem. The Stern algorithm
can be generalized to  handle $p$-spin problems with arbitrary couplings as
well.

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
