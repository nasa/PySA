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
from typing import Tuple
from itertools import combinations
from math import comb

from mceliece.f2mat import f2_matmul
from mceliece.gausselm import standardize_parcheck, gauss_jordan_unpacked
from nistrng import unpack_sequence


def hw_table(n, t):
    nct = comb(n, t)
    tbl = np.zeros((nct, t), dtype=int)
    for i, tbits in enumerate(combinations(list(range(n)), t)):
        tbl[i, :] = np.asarray(tbits, dtype=int)
    return tbl


def enumerate_syndromes(H: np.ndarray, t):
    n = H.shape[1]
    num_errs = sum(comb(n, w) for w in range(t + 1))
    err_bits = np.zeros((num_errs, n), dtype=np.int8)
    i = 0
    for w in range(t + 1):
        for tbits in combinations(list(range(n)), w):
            for tt in tbits:
                err_bits[i, tt] = 1
            i += 1
    syndromes = f2_matmul(err_bits, H.transpose())  # [num_errs, s]
    synd_dict = {
        tuple(np.packbits(synd)): err for err, synd in zip(err_bits, syndromes)
    }
    return synd_dict


def dual_code(G: np.ndarray) -> np.ndarray:
    """

    :param G: (k, n) code matrix
    :return: (n-k, n) dual code matrix
    """
    k = G.shape[0]
    n = G.shape[1]
    H = np.zeros((n - k, n), dtype=np.int8)
    G2, r, pivs = gauss_jordan_unpacked(G, upto=k)
    if r != k:
        raise RuntimeError(f"Expected code rank r = {k}. Found {r}.")
    non_pivs = np.sort(np.asarray(list(set(range(n)) - set(pivs))))
    for i, p in enumerate(non_pivs):
        H[i, p] = 1
    for i, col in enumerate(pivs):
        H[:, col] = G[i, non_pivs]

    return H


class LinearCode:

    def encode(self, m: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def correct_err(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @property
    def G(self) -> np.ndarray:
        """
        The k x n encoding matrix of the code as an unpacked int8 array
        :return:
        """
        raise NotImplementedError

    @property
    def H(self) -> np.ndarray:
        """
        The (n-k) x n parity check matrix of the code as an unpacked int8 array
        :return:
        """
        raise NotImplementedError

    @property
    def n(self) -> int:
        raise NotImplementedError

    @property
    def k(self) -> int:
        raise NotImplementedError
