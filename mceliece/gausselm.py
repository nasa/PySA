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


@numba.njit
def find_pivot_u8(M, col):
    """
    Get the index of the first row that is available as a pivot for the given column.
    :param M:
    :param col:
    :return:
    """
    n = M.shape[0]
    i_ = col // 8
    j_ = (col % 8)
    msk = np.dtype('uint8').type(1) << j_
    pivs = np.not_equal(np.bitwise_and(M[:, i_], msk),
                        np.dtype('uint8').type(0))
    lo_pivs = np.nonzero(
        np.logical_and(pivs, np.greater_equal(np.arange(n), col)))[0]
    up_pivs = np.nonzero(np.logical_and(pivs, np.less(np.arange(n), col)))[0]
    return lo_pivs, up_pivs


@numba.njit
def find_pivot_unpacked(M, col, row):
    """
    Get the index of the first row that is available as a pivot for the given column.
    :param M:
    :param col:
    :return:
    """
    n = M.shape[0]
    pivs = np.not_equal(np.bitwise_and(M[:, col], 1), 0)
    lo_pivs = np.nonzero(
        np.logical_and(pivs, np.greater_equal(np.arange(n), row)))[0]
    up_pivs = np.nonzero(np.logical_and(pivs, np.less(np.arange(n), row)))[0]
    return lo_pivs, up_pivs


@numba.njit
def swap_rows(M, i1, i2):
    r = np.copy(M[i1, :])
    M[i1, :] = M[i2, :]
    M[i2, :] = r
    return M


@numba.njit
def add_row_to(M, i1, i2):
    M[i2, :] ^= M[i1, :]
    return M


@numba.njit
def gauss_jordan(M, abort_noninv=False):
    """
    Perform row-wise Gauss-Jordan elimination on the F2 matrix M in-place.
    M should be a bit-packed matrix and should be augmented to keep track of row operations.
    M is reduced to row echelon form and its rank r is determined.
    In addition, if M is full rank, the main block is reduced to the identity matrix and the
    augmented block will be left-multiplied by M^-1.

    :param M:
    :return:
    """
    n = M.shape[0]
    n2 = M.shape[1]  # bit-packed, possibly augmented dimension
    if n2 * 8 < n:  # invalid dimensions
        return M, -1
    r = 0  # rank
    for i in range(n):  # for each column i
        lo_pivs, up_pivs = find_pivot_u8(M, i)  # find candidate pivot rows
        if len(lo_pivs) == 0:  # Continue or terminate if no pivots.
            if abort_noninv:
                break
            else:
                continue
        r += 1  # Increment the rank
        pi = lo_pivs[0]
        if pi != i:  # swap the pivot row if necessary
            swap_rows(M, pi, i)
        lo_pivs = lo_pivs[
            1:]  # remaining lower rows that are non-zero on this column
        mi = M[i, :]
        # eliminate rows
        for i2 in lo_pivs:
            M[i2, :] ^= mi
        for i2 in up_pivs:
            M[i2, :] ^= mi

    return M, r


@numba.njit
def gauss_jordan_unpacked(M, abort_noninv=False, upto=None, rre=True):
    """
    Perform row-wise Gauss-Jordan elimination on the F2 matrix M in-place.
    M should be a bit-packed matrix and should be augmented to keep track of row operations.
    M is reduced to row echelon form and its rank r is determined.
    In addition, if M is full rank, the main block is reduced to the identity matrix and the
    augmented block will be left-multiplied by M^-1.

    :param M:
    :return:
    """
    n = M.shape[0]
    n2 = M.shape[1]  # possibly augmented dimension
    r = 0  # rank
    if upto is None:
        upto = n
    piv_cols = []
    for i in range(n2):  # for each column i
        if r == upto:
            break
        lo_pivs, up_pivs = find_pivot_unpacked(M, i,
                                               r)  # find candidate pivot rows
        if len(lo_pivs) == 0:  # Continue or terminate if no pivots.
            if abort_noninv:
                break
            else:
                continue
        piv_cols.append(i)
        pi = lo_pivs[0]
        if pi != r:  # swap the pivot row if necessary
            swap_rows(M, pi, r)
        lo_pivs = lo_pivs[
            1:]  # remaining lower rows that are non-zero on this column
        # eliminate rows
        for i2 in lo_pivs:
            add_row_to(M, r, i2)
        if rre:  # attempt reduced row echelon form
            for i2 in up_pivs:
                add_row_to(M, r, i2)
        r += 1  # Increment the rank

    return M, r, piv_cols


def null_space(H: np.ndarray):
    if len(H.shape) != 2:
        raise ValueError("Expected a matrix in null_space")
    Ht = H.transpose()
    (n, m) = Ht.shape
    Haug = np.concatenate([Ht, np.eye(n, dtype=np.int8)], axis=1)
    _, r, _ = gauss_jordan_unpacked(Haug, upto=m)
    Hgj = Haug.transpose()
    B = Hgj[:m, :]  # [m, n]
    C = Hgj[m:, :]  # [n, n]
    Bzcols = np.all(B == 0, axis=0) > 0  # all-zero columns of B
    Cnzcols = np.any(np.asarray(C != 0, dtype=np.int8),
                     axis=0) > 0  # non-zero columns of B
    # Columns of C that span the null space
    nullsp_cols = np.nonzero(np.logical_and(Bzcols, Cnzcols))[0]
    G = C[:, nullsp_cols]
    return G


def standardize_parcheck(H: np.ndarray):
    """
    Standardize an unpacked parity check matrix into the form H = (A | I_{n-k} )

    [ A | I_{n-k}] = S H P

    :param H: (n-k, n) parity check matrix
    :return: the (n-k, k) matrix A
    """
    nmk = H.shape[0]
    n = H.shape[1]
    # First reduce H to row echelon form
    Haug = np.concatenate([H, np.eye(nmk, dtype=np.int8)],
                          axis=1)  # [nmk, n + nmk]
    H1, _, pivs = gauss_jordan_unpacked(Haug,
                                        abort_noninv=False,
                                        upto=nmk,
                                        rre=False)
    non_pivs = np.sort(np.asarray(list(set(range(n + nmk)) - set(pivs))))
    # Permute the columns of H so that the pivots are the first n-k columns
    P = np.concatenate([pivs, non_pivs])
    H2 = H1[:, P]
    # reduce with gauss-jordan again
    H3, r = gauss_jordan_unpacked(H2, abort_noninv=False, upto=nmk, rre=True)
    S = H3[:, n:]
    if r < nmk:
        raise ValueError(
            f"Parity check matrix could not be standardized. (Rank {r} < {nmk})"
        )
    A = H3[:, nmk:n]
    return A, P, S
