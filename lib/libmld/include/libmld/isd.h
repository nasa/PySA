/*
Author: Humberto Munoz Bauza (humberto.munozbauza@nasa.gov)

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
*/

#ifndef LIBMLD_ISD_H
#define LIBMLD_ISD_H

#include "bitvec.h"
#include "gauselm.h"
#include <algorithm>
#include <cstdint>
#include <random>

typedef std::mt19937_64 stern_rng_t;

enum SternColType : uint8_t {
  None = 0,
  Redundancy = 1,
  InfoSet = 2,
  Permanent = 4,
  Auxiliary = 8
};

template <typename T> class SternP1 {
  /// Stern's algorithm (p=1)
  /// Parameters:
  /// - l number of bits from the redundancy set to collide
  /// - p = 1size of linear combinations in each split
  /// - m number of l-combinations to check for collisions
public:
  SternP1(BitMatrix<T> &G, uint32_t n, uint32_t k, stern_rng_t rng)
      : G(G), work_vec(G.rows), work_vec_i(G.rows), solution_vec(n),
        work_vec_row(k), k(k), n(n), workn(n), pivt_cols(n),
        col_types(n, SternColType::None), rng(rng) {
    if (G.cols < n + 1) {
      throw std::invalid_argument("Invalid number of columns in G.");
    }
    if (G.rows != n - k) {
      throw std::invalid_argument("Invalid number of rows in G.");
    }
    info_set.reserve(k);
  }

  void sample_ir_split() {
    /// Randomly split the columns of G into an information/redundancy set split
    for (uint32_t i = 0; i < n; ++i) {
      workn[i] = i;
    }
    std::shuffle(workn.begin(), workn.end(), rng);
    uint64_t rank = gaussian_elimination(G, workn, pivt_cols);
    if (rank != G.rows) {
      throw std::runtime_error("Invalid rank for G");
    }
    std::fill(col_types.begin(), col_types.end(), SternColType::InfoSet);
    for (uint32_t j : pivt_cols) {
      col_types[j] = SternColType::Redundancy;
    }
    info_set.clear();
    for (uint32_t i = 0; i < n; ++i) {
      if (col_types[i] == SternColType::InfoSet) {
        info_set.push_back(i);
      }
    }
    if (info_set.size() != k) {
      throw std::runtime_error("Invalid information set size");
    }
  }

  void shuffle_isd() { std::shuffle(info_set.begin(), info_set.end(), rng); };

  bool collision_iteration_heavy(uint32_t w) {
    uint32_t k1 = k / 2;
    uint32_t k2 = k - k1;
    size_t num_blocks = G.row_blocks;
    T *Gpn = &G.get_block(0, n);
    T *workpi = &work_vec_i.get_block(0);
    for (uint32_t i = 0; i < k1; ++i) {
      size_t coli = info_set[i];
      T *Gpi = &G.get_block(0, coli);
#if defined(__INTEL_COMPILER)
      __assume(num_blocks % alnb == 0);
      __assume_aligned(Gpi, alignment);
      __assume_aligned(Gpn, alignment);
      __assume_aligned(workpi, alignment);
#endif
      for (size_t bi = 0; bi < num_blocks; ++bi)
        workpi[bi] = Gpi[bi] ^ Gpn[bi];

      for (uint32_t j = 0; j < k2; ++j) {
        size_t colj = info_set[k1 + j];
        T *Gpj = &G.get_block(0, colj);
#if defined(__INTEL_COMPILER)
        __assume(num_blocks % alnb == 0);
        __assume_aligned(Gpi, alignment);
        __assume_aligned(Gpn, alignment);
        __assume_aligned(Gpj, alignment);
#endif
        // Evaluate hamming distance from y
        size_t hw = 0;
        for (size_t bi = 0; bi < num_blocks; ++bi)
          hw += BitVecNums<T>::popcount(workpi[bi] ^ Gpj[bi]);

        if (hw > w - 2 * p) { // Hamming weight too large
          continue;
        } else {
          finalize_solution(i, k1 + j);
          return true;
        }
      }
    }
    return false;
  }

  template <typename U = uint8_t, bool testhw1 = false>
  bool collision_iteration(uint32_t w) {
    for (uint32_t i = 0; i < k; ++i)
      work_vec_row[i] = G.get_block(0, info_set[i]);
    uint32_t k1 = k / 2;
    uint32_t k2 = k - k1;
    size_t num_blocks = G.row_blocks;
    T *Gpn = &G.get_block(0, n);
    T y_row = G.get_block(0, n);
    for (uint32_t j = 0; j < k2; ++j) {
      T _tmpi = work_vec_row[k1 + j] ^ y_row;
      for (uint32_t i = 0; i < k1; ++i) {
        T _tmp = work_vec_row[i] ^ _tmpi;
        bool _tst;
        if constexpr (testhw1) {
          _tst = nz2kpopblk<T, U>(_tmp);
        } else {
          _tst = nzpopblk<T, U>(_tmp);
        }
        if (_tst) {
          size_t colj = info_set[k1 + j];
          T *Gpj = &G.get_block(0, colj);
          size_t coli = info_set[i];
          T *Gpi = &G.get_block(0, coli);
#if defined(__INTEL_COMPILER)
          __assume(num_blocks % alnb == 0);
          __assume_aligned(Gpi, alignment);
          __assume_aligned(Gpn, alignment);
          __assume_aligned(Gpj, alignment);
#endif
          size_t hw = 0;
          for (size_t bi = 0; bi < num_blocks; ++bi) {
            T _tmpij = Gpi[bi] ^ Gpj[bi];
            hw += BitVecNums<T>::popcount(_tmpij ^ Gpn[bi]);
          }
          if (hw <= w - 2 * p) { // Hamming weight reached
            finalize_solution(i, k1 + j);
            return true;
          }
        }
      }
    }
    return false;
  }

  const BitVec<T> &get_solution_vec() const { return solution_vec; }

protected:
  void finalize_solution(size_t isd1, size_t isd2) {
    auto Gi = G.column_slice(info_set[isd1]);
    auto Gj = G.column_slice(info_set[isd2]);
    for (size_t bi = 0; bi < work_vec.num_blocks(); ++bi) {
      work_vec.get_block(bi) =
          Gi.get_block(bi) ^ Gj.get_block(bi) ^ G.get_block(bi, n);
    }
    // The solution has the desired hamming weight with w-2p errors in the
    // redundancy set
    solution_vec.clear();
    for (uint32_t q = 0; q < G.rows; ++q) {
      if (work_vec(q)) {
        solution_vec.set(pivt_cols[q], true);
      }
    }
    solution_vec.set(info_set[isd1], true);
    solution_vec.set(info_set[isd2], true);
  }
  BitMatrix<T> G;
  BitVec<T> work_vec;
  BitVec<T> work_vec_i;
  BitVec<T> solution_vec;
  std::vector<T> work_vec_row;
  std::vector<T> work_vec_row2;
  uint32_t k, n;
  uint32_t p = 1;
  std::vector<size_t> workn;
  std::vector<size_t>
      pivt_cols; // Pivot columns of gaussian elimination == redundancy set
  std::vector<SternColType> col_types;
  std::vector<uint32_t> info_set;
  stern_rng_t rng;

  static const size_t alignment = BitVecNums<T>::alignment;
  static const size_t alnb = BitVecNums<T>::aligned_num_blocks;
};

#endif // LIBMLD_ISD_H
