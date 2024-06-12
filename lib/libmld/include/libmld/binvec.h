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

//! Fairly basic binary vector and matrix classes with each bit encoded within
//! one byte

#ifndef LIBMLD_BINVEC_H
#define LIBMLD_BINVEC_H
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

class BinMatC {
  //! Binary matrix with C-ordered storage. To be yeeted.
public:
  BinMatC(size_t n, size_t m) : n(n), m(m), vec(n * m, 0) {}
  std::tuple<size_t, size_t> size() const { return {n, m}; }

  uint8_t &operator()(size_t i, size_t j) {
    if (i < n && j < m)
      return vec[i * m + j];
    else
      throw std::runtime_error("BinaryMat indices out of bounds.");
  }

  const uint8_t &operator()(size_t i, size_t j) const {
    if (i < n && j < m)
      return vec[i * m + j];
    else
      throw std::runtime_error("BinaryMat indices out of bounds.");
  }

  BinMatC &operator+=(const BinMatC &other) {
    if (std::tie(n, m) != other.size()) {
      throw std::runtime_error("Invalid BinaryMat dimensions.");
    }
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        vec[i * m + j] ^= other.vec[i * m + j];
      }
    }
    return *this;
  }
  friend std::ostream &operator<<(std::ostream &os, const BinMatC &matrix) {
    size_t nbytes = 1 + matrix.n / 8 - (matrix.n % 8 == 0 ? 1 : 0);
    std::ios::fmtflags os_flags(os.flags());
    os << std::setfill('0') << std::hex;

    for (size_t j = 0; j < matrix.m; ++j) {
      for (size_t b = 0; b < nbytes; ++b) {
        uint8_t bt = 0;
        for (size_t ii = 0; ii < 8; ++ii) {
          size_t i = 8 * b + ii;
          if (i < matrix.n) {
            bt |= (matrix(i, j) > 0 ? 1 : 0) << ii;
          }
        }
        os << std::setw(2) << (uint16_t)bt << ' ';
      }
      os << '\n';
    }
    os.flags(os_flags);
    return os;
  }

  uint8_t *data() { return vec.data(); }

private:
  size_t n;
  size_t m;
  std::vector<uint8_t> vec;
};

#endif // LIBMLD_BINVEC_H
