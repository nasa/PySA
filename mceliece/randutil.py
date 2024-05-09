# Author: Humberto Munoz Bauza (humberto.munozbauza@nasa.gov)
#
# Copyright © 2023, United States Government, as represented by the Administrator
# of the National Aeronautics and Space Administration. All rights reserved.
#
# The PySA, a powerful tool for solving optimization problems is licensed under
# the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from mceliece.f2mat import invert_triangular, packed_eye, F2Matrix, f2_matmul
from mceliece.gausselm import gauss_jordan, gauss_jordan_unpacked


def random_weight_t(n, t, rng: np.random.Generator = None):
    """
    Generate a random bit vector of length n and Hamming weight t
    :param n:
    :param rng:
    :return:
    """
    if rng is None:
        rng = np.random.default_rng()
    arr = np.zeros(n, dtype=np.int8)
    idxs = rng.choice(n, t, replace=False)
    arr[idxs] = 1
    return arr


def random_perm_f2(n, rng: np.random.Generator = None):
    """
    Generate a random n x n permutation matrix over F2
    """
    if rng is None:
        rng = np.random.default_rng()
    idxs = np.arange(n)
    rng.shuffle(idxs)
    p = np.zeros((n, n), dtype=np.int8)
    for i, j in zip(idxs, range(n)):
        p[i, j] = 1

    return p


def random_invertible_f2(n, rng: np.random.Generator = None):
    """
        Generate a random n x n invertible matrix over the field F2.
        It is asymptotically guaranteed that a random matrix over GF(2) is invertible with probability ~0.26
        """
    for i in range(100):
        if rng is None:
            rng = np.random.default_rng()

        r = rng.integers(0, 2, size=(n, n), dtype=np.int8)
        raug = np.concatenate([r, np.eye(n, dtype=np.int8)], axis=1)
        relim, rank, _ = gauss_jordan_unpacked(raug, abort_noninv=True)
        if rank == n:
            rinv = relim[:, n:]
            #r = F2Matrix(r)
            #rinv = F2Matrix(rinv)
            return r, rinv

    raise RuntimeError("Failed to generate random full-rank matrix.")


def random_invertible_packed_f2(n, rng: np.random.Generator = None):
    """
        Generate a bit-packed random n x n invertible matrix over the field F2.
        It is asymptotically guaranteed that a random matrix over GF(2) is invertible with probability ~0.26
        """
    for i in range(100):
        if rng is None:
            rng = np.random.default_rng()
        n2 = n // 8

        r = rng.integers(0, 256, size=(n, n2), dtype=np.uint8)
        raug = np.concatenate([r, packed_eye(n)], axis=1)
        relim, rank = gauss_jordan(raug, abort_noninv=True)
        if rank == n:
            rinv = relim[:, n2:]
            r = F2Matrix(r)
            rinv = F2Matrix(rinv)
            return r, rinv

    raise RuntimeError("Failed to generate random full-rank matrix.")
