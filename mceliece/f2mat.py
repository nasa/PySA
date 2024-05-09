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
import numba


def packed_eye(n):
    m = np.eye(n, dtype=np.int8)
    m_packed = np.packbits(m, axis=1, bitorder='little')
    return m_packed


@numba.njit
def _f2_matmul_jit(a: np.ndarray, b: np.ndarray):
    d1, d2 = a.shape
    d2b, d3 = b.shape
    c = np.zeros((d1, d3), dtype=np.int8)
    for i in range(d1):
        for j in range(d3):
            cij = 0
            for k in range(d2):
                cij ^= (a[i, k] & b[k, j]) & 1
            c[i, j] = cij
    return c


def f2_matmul(a: np.ndarray, b: np.ndarray):
    """
    Multiply two unpacked matrices over F2
    """
    d1, d2 = a.shape
    d2b, d3 = b.shape
    if d2 != d2b:
        raise ValueError("Shape Mismatch")
    return _f2_matmul_jit(a, b)


def f2_sum_to(iterable, out: np.ndarray):
    """
    add an iterable of arrays to out over F2
    """
    tmp = np.zeros_like(out, dtype=int)
    for m in iterable:
        tmp += m
    out[:] += tmp[:] % 2


def invert_triangular(L: np.ndarray):
    """
        Invert the linearly independent lower or upper triangular matrix L over F2
        L should be passed as an unpacked n x n matrix
        Implements a simplification of
         https://math.stackexchange.com/questions/1003801/inverse-of-an-invertible-upper-triangular-matrix-of-order-3
    """
    n = L.shape[0]
    assert len(L.shape) == 2 and L.shape[0] == L.shape[1]
    # strict upper/lower triangle
    T = np.copy(L)
    for i in range(n):
        T[i, i] = 0
    Tpows = np.zeros((n, n, n), dtype=np.int8)
    negT = T  # -T == T in this field
    Tpows[0, :, :] = np.eye(n, dtype=np.int8)  # -T
    for k in range(1, n):
        Tpows[k, :, :] = f2_matmul(negT, Tpows[k - 1, :, :])
    Tinv = np.zeros((n, n), dtype=np.int8)
    f2_sum_to([Tpows[i, :, :] for i in range(n)], Tinv)
    return Tinv


class F2Vec:

    def __init__(self, v):
        v = np.asarray(v)
        self._n = len(v)
        if v.dtype == np.uint8:
            self._v = v
        else:
            self._v = np.packbits(v)


class F2Matrix:

    def __init__(self, m: np.ndarray):
        if m.dtype == np.uint8:
            self._m = m
            self._dims = (m.shape[0], m.shape[1] * 8)
        else:
            self._dims = np.shape(m)
            self._m = np.packbits(m, axis=-1)

    def __matmul__(self, other):
        """
        Perform matrix multiplication
        The inner dimension for an F2 Matrix is always the second dimension,
        so  "A @ B" actually evaluates  $A B^T$.
        The matrix return is an unpacked matrix
        """
        if isinstance(other, F2Vec):
            if self._dims[1] == other._n:
                a = np.bitwise_and(self._m, other._v[np.newaxis, :])
                bitsum = sum(
                    np.bitwise_and(np.right_shift(a, i), 1) for i in range(8))
                bitsum = np.sum(bitsum, axis=-1, dtype=int)
                par = bitsum % 2
                return F2Vec(par)
            else:
                raise ValueError("Dimension mismatch")
        elif isinstance(other, F2Matrix):
            # The inner dimension is always the second dimension for an F2Matrix
            if self._dims[1] == other._dims[1]:
                a = np.bitwise_and(self._m[:, np.newaxis, :],
                                   other._m[np.newaxis, :, :])
                bitsum = sum(
                    np.bitwise_and(np.right_shift(a, i), 1) for i in range(8))
                bitsum = np.sum(bitsum, axis=-1, dtype=int)
                par = bitsum % 2
                return par.astype(np.int8)
            else:
                raise ValueError("Dimension Mismatch")
