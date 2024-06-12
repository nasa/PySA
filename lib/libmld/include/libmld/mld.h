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

#ifndef LIBMLD_MLD_H
#define LIBMLD_MLD_H

#include "libmld/binvec.h"
#include "libmld/bitvec.h"
#include <string>
#include <vector>

typedef uint32_t mld_uint_t;

class MLDException : public std::exception {
private:
  const char *message;

public:
  explicit MLDException(const char *msg) : message(msg) {}
  explicit MLDException(const std::string &msg) : message(msg.c_str()) {}
  const char *what() const noexcept { return message; }
};

enum MLDType { G, H, P, W };

class MLDProblem {
public:
  // MLDProblem(uint8_t y=0) : y(y), nvars(0), nrows(0) {};
  MLDProblem() = default;
  MLDProblem(const MLDProblem &) = default;
  void read_problem(std::istream &input_stream);
  [[nodiscard]] MLDType problem_type() const { return prob_type; }
  [[nodiscard]] size_t NVars() const { return nvars; }
  [[nodiscard]] size_t NClauses() const { return nrows; }
  bool is_parcheck() { return prob_type == MLDType::H; }
  size_t CodeLength() {
    /// Length n of the code
    if (prob_type == MLDType::H) {
      return nvars;
    } else {
      return nrows;
    }
  }
  size_t CodeDim() {
    /// Dimension k of the code
    if (prob_type == MLDType::H) {
      return nvars - nrows;
    } else {
      return nvars;
    }
  }
  [[nodiscard]] int64_t Weight() const { return w; }

  [[nodiscard]] const std::vector<std::vector<mld_uint_t>> &
  clause_list() const {
    return clauses;
  }

  [[nodiscard]] const std::vector<uint8_t> &yenc() const { return yarr; }

  template <typename T> BitMatrix<T> clauses_as_bitmatrix(bool y_aug = true) {
    BitMatrix<T> bm(nrows, nvars + (y_aug ? 1 : 0));
    for (size_t i = 0; i < nrows; ++i) {
      for (mld_uint_t j : clauses[i]) {
        bm.set(i, j, true);
      }
      if (y_aug && yarr[i] > 0)
        bm.set(i, nvars, true);
    }
    return bm;
  }

  BinMatC clauses_as_binmat(bool y_aug = true) {
    BinMatC bm(nrows, nvars + (y_aug ? 1 : 0));
    for (size_t i = 0; i < nrows; ++i) {
      for (mld_uint_t j : clauses[i]) {
        bm(i, j) = 1;
      }
      if (y_aug && yarr[i] > 0)
        bm(i, nvars) = 1;
    }
    return bm;
  }

  template <typename T> BitVec<T> y_as_bitvec() {
    BitVec<T> yv(nrows);
    for (size_t i = 0; i < nrows; ++i) {
      if (yarr[i] > 0)
        yv.set(i, true);
    }
    return yv;
  }

private:
  MLDType prob_type{MLDType::G};
  size_t nvars = 0;
  size_t nrows = 0;
  std::vector<uint8_t> yarr{};
  std::vector<std::vector<mld_uint_t>> clauses{};
  uint16_t y = 0; // default values of the y vector
  int64_t w = 0;  // weight of the problem
};

#endif // LIBMLD_MLD_H
