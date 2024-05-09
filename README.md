# McEliece Instance Generator

**Disclaimer**: This package is for research and testing purposes only. 
It is not to be used for encrypting sensitive data.

The Python application `genmci` generates random McEliece and Niederreider problem 
instances with Goppa codes, consisting of a public key binary matrix and a ciphertext binary vector.
The problem instances can be generated for any code size `n` and minimum error correction distance `t`.

```
NAME
    genmci - Generates McEliece Cryptographic Instances (MCI).

SYNOPSIS
    genmci N T NUM_INST PUB_DIR PRIV_DIR <flags>

DESCRIPTION
    Generates McEliece Cryptographic Instances (MCI).

POSITIONAL ARGUMENTS
    N
        Type: int
        Code length.
    T
        Type: int
        Number of correctable errors (Goppa polynomial degree).
    NUM_INST
        Type: int
    PUB_DIR
        Type: str
        Public destination directory.
    PRIV_DIR
        Type: str
        Private destination directory.

FLAGS
    --file-pattern=FILE_PATTERN
        Type: str
        Default: ${T}_n${n}_t${t}_m${m}_k${k}
        Pattern to use to create files.
    --f2-deg=F2_DEG
        Type: Optional[int]
        Default: None
        Extension field degree. Defaults to the smallest degree that
        will contain a code of size n.
    -n, --nieder=NIEDER
        Type: bool
        Default: False
        Generate McEliece-Niederreiter instances instead of McEliece
        instances.
    -s, --seed=SEED
        Type: Optional[int]
        Default: None
        Seed for the random number generator. Will seed non-deterministically
        if not provided.
```

An MCI problem instance consists of a binary matrix G and a ciphertext binary vector y.
The solution to each instance is a plaintext binary vector x such that the objective function  $L(x) = ||y - Gx||_1$  is minimized.
        
Equivalently a McEliece-Niederreiter Cryptographic Instance (MNCI) consists of a binary matrix H 
and a binary vector s. The solution to each instance is a binary vector y of minimum Hamming weight 
(or known Hamming weight t) such that  s = H y. 
MCI can be reduced to MNCI and vice-versa. They are equivalent formulations of the linear decoding problem,
which is NP-complete in general.
        
Each instance is split into public and private sets of files.
    Private:
        - The message that was encoded (plain.txt)
        - The Goppa code used for encoding (goppa.yml)
        - The random matrices of the McEliece protocol (.priv.txt)
    Public:
        - The encoded message (cipher.txt)
        - The public key matrix (pub.txt)
        - The native decoding problem in MLD (i.e. p-spin) format (mld.txt)
        - The native decoding problem in XOR-extension DIMACS format (xor.cnf)
        - The decoding problem reduced to 3-SAT in DIMACS format (3sat.cnf)
        
* Maximum likelihood decoding (MLD) format *
The MLD problem is specified as follows: Given an M x N matrix H, an M-vector y, and a positive integer w, 
find the N-vector x such that
            Hx = y (mod 2)
subject to the constraint  |x| <= w (or |x| = w)

The MLD file format describes either a MLD or PCD problem using a CNF-like specification, where
each line describes a row of the linear equation over F2.
The overall format is summarized as follows:
```txt
     (g|h) N M y w
     1 2 3 4
    -2 3 4 5
     ...
```

How to install
---

The easiest way to install `genmci` is using `pip`:
```
pip install git+https://github.com/nasa/pysa@pysa-mceliece
```

MLD file format
----

The MLD file format describes a maximum likelihood decoding problem using a CNF-like specification, where each line describes a row of the linear equation over $\mathbb{F}_2$. The overall format is summarized as follows:

```txt
(g|h) N M y w
 1 2 3 4
-2 3 4 5
...
```

The first line is a header with the following data:

- A single character, `g` or `h`
      - If `g`, the file describes the $n\times k$ generator matrix transpose for a $[n, k]$ code with `M = n` clauses and `N = k` variables.
      - If `h`, the file describes the parity check matrix for a $[n, k]$ code with `M = n-k` clauses and `N = n` variables.
      - The characters `p` and `w` are reserved.
- `N` Number of column variables.
- `M` Number of rows, i.e. clauses.
- `y = (0|1)` Sets the boolean assignment for each row to either false (`0`) or true (`1`).
- `w` Hamming weight of the code space error.
      - `w > 0` Hamming weight is exactly `w`
      - `w == 0` Hamming weight is not specified
      - `w < 0` Hamming weight is at most `|w|`

Lines 2 through `M+1` specify the non-zero elements of the matrix using 1-based indices (1 to `N`). If an integer is negative, then $y_i$ is negated to $(1 - y_i)$.

As an example, the Niederreiter instances contain the header `h n M 1 t`, where `M=n-k` and `t` is the error correction distance of the Goppa decoder. Negated lines correspond to rows where $y_i = 0$, and other lines correspond to rows where $y_i = 1$.

*p*-spin Hamiltonian Ground State
----

A *p*-spin Hamiltonian over $k$ variables has the form

$$ H = -\sum_{i=1}^N J_i \sigma_{i_1} \ldots \sigma_{i_p}.  $$

When $J_i=\pm 1$, the problem of finding the ground state of this Hamiltonian can be reduced to a maximum likelihood decoding problem.


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
