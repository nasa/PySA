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

#include <sstream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "tseytin.h"

namespace tseytin {

void add_weight_constraint(int t, XORSatClauses<Lit> &solver, int n) {
  if (n < 0)
    n = solver.nVars();
  TseytinSAT tseytin(solver.nVars());
  if (t > 0) {
    std::vector<Lit> lits;
    lits.resize(n);
    for (int i = 0; i < n; ++i)
      lits[i] = Lit(i, false);
    tseytin.accumulate(lits, t);
    unsigned int num_all_vars = tseytin.current_num_vars;
    // std::cout << "Additional clauses for weight constraint: " <<
    // tseytin.extra_clauses.size() << '\n'; std::cout << "Additional variables
    // for weight constraint: " << num_all_vars - n << '\n';
    solver.new_vars(num_all_vars - n);

    for (const std::vector<Lit> &cl : tseytin.extra_clauses) {
      solver.add_clause(cl);
    }
  }
}

void into_3sat_only(XORSatClauses<Lit> &solver) {
  unsigned int n = solver.n_vars;
  unsigned int m = solver.xor_rhs.size();
  unsigned int aux_var = n;
  std::map<std::tuple<int32_t, int32_t>, int32_t> aux_tbl;
  std::vector<std::tuple<int32_t, int32_t>> reduced_xor;
  reduced_xor.reserve(m);

  for (int32_t i = 0; i < m; ++i) {
    auto &Ci = solver.xor_rows[i];
    size_t ki = Ci.size();
    if (ki > 1) {
      int32_t u = Ci[0];
      int32_t v = Ci[1];
      if (ki > 2) {
        for (int32_t j = 2; j < ki; ++j) {
          if (auto it = aux_tbl.find(std::make_tuple(u, v));
              it != aux_tbl.end()) {
            u = it->second;
            v = Ci[j];
          } else {
            aux_tbl[std::make_tuple(u, v)] = aux_var;
            u = aux_var;
            v = Ci[j];
            ++aux_var;
          }
        }
      }
      // clause is reduced to u xor v
      reduced_xor.emplace_back(u, v);
    } else {
      reduced_xor.emplace_back(Ci[0], -1);
    }
  }
  uint32_t new_vars = aux_var - n;
  solver.new_vars(new_vars);
  for (int32_t i = 0; i < m; ++i) {
    auto ci = reduced_xor[i];
    int32_t x1, x2;
    x1 = std::get<0>(ci);
    x2 = std::get<1>(ci);
    uint8_t &rhs = solver.xor_rhs[i];
    if (x2 >= 0) {
      solver.add_clause({Lit(x1, rhs == 0), Lit(x2, false)});
      solver.add_clause({Lit(x1, rhs != 0), Lit(x2, true)});
    } else {
      solver.add_clause({Lit(x1, rhs == 0)});
    }
  }
  for (const auto &it : aux_tbl) {
    solver.add_clause({Lit(std::get<0>(it.first), false),
                       Lit(std::get<1>(it.first), false),
                       Lit(it.second, true)});
    solver.add_clause({Lit(std::get<0>(it.first), false),
                       Lit(std::get<1>(it.first), true),
                       Lit(it.second, false)});
    solver.add_clause({Lit(std::get<0>(it.first), true),
                       Lit(std::get<1>(it.first), false),
                       Lit(it.second, false)});
    solver.add_clause({Lit(std::get<0>(it.first), true),
                       Lit(std::get<1>(it.first), true), Lit(it.second, true)});
  }
  solver.xor_rhs.clear();
  solver.xor_rows.clear();
}

XORSatClauses<Lit>
xorsat_from_numpy_arrays(py::array_t<int8_t, py::array::c_style> &G,
                         py::array_t<int8_t, py::array::c_style> &y) {
  XORSatClauses<Lit> xs;
  py::buffer_info G_buf = G.request();
  py::buffer_info y_buf = y.request();

  // check shapes
  if (G_buf.ndim != 2 && y_buf.ndim != 1)
    throw std::runtime_error("Expected a 2D array for G and a 1D array for y.");
  // check size consistency
  size_t nrows = G_buf.shape[0];
  size_t ncols = G_buf.shape[1];
  if (nrows != y_buf.shape[0])
    throw std::runtime_error("Number of G columns does not match length of y.");
  // Convert G and y into a pure xorsat problem
  xs.n_vars = ncols;
  auto garr = G.unchecked<2>();
  auto yarr = y.unchecked<1>();
  std::vector<uint32_t> xor_row;
  xor_row.reserve(ncols);
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = 0; j < ncols; ++j) {
      if (garr(i, j) > 0)
        xor_row.push_back(j);
    }
    xs.add_xor_clause(xor_row, yarr(i));
    xor_row.clear();
  }

  return xs;
}
} // namespace tseytin

using namespace tseytin;

PYBIND11_MODULE(bindings, m) {
  py::class_<XORSatClauses<Lit>>(m, "XOrSatClauses")
      .def("nvars", &XORSatClauses<Lit>::nVars)
      .def("as_dimacs_str", [](const XORSatClauses<Lit> &self) {
        std::stringstream oss;
        self.write_dimacs(oss);
        return oss.str();
      });
  m.def("xorsat_from_numpy_arrays", &xorsat_from_numpy_arrays);
  m.def("add_weight_constraints", &add_weight_constraint);
  m.def("into_3sat_only", &into_3sat_only);
}