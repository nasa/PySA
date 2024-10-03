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

#include <fstream>

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pysa/sat/walksat.hpp"
namespace py = pybind11;

struct WalksatResultState{
  std::vector<uint8_t> state;
};

struct WalksatResult{
  WalksatResultState result_state;
  uint64_t num_unsat = 0;
  size_t preproc_time_us = 0;
  size_t computation_time_us = 0;
  size_t iterations = 0;
};

WalksatResult walksat_optimize_core(
    const pysa::sat::Formula_WS& formula_ws,
    uint64_t max_steps,
    double p=0.5,
    uint64_t max_unsat=0,
    uint64_t random_seed=0){

  pysa::sat::WalkSatOptimizer wsopt(formula_ws, random_seed, p);
  wsopt.restart_state();
  size_t i = 0;
  uint64_t n_unsat = formula_ws.num_clauses();
  auto it_ = std::chrono::high_resolution_clock::now();
  for (; i < max_steps; ++i) {
    n_unsat = wsopt.step();
    if (n_unsat <= max_unsat)
      break;
  }
  auto et_ = std::chrono::high_resolution_clock::now();
  WalksatResult result;
  result.result_state = WalksatResultState{std::move(wsopt.state())};
  result.num_unsat = n_unsat;
  result.computation_time_us = std::chrono::duration_cast<std::chrono::microseconds>(et_ - it_)
      .count();
  result.iterations = i;
  return result;
}

WalksatResult walksat_optimize_py(
    const std::vector<std::vector<int32_t>>& formula,
    uint64_t max_steps,
    double p=0.5,
    uint64_t max_unsat=0,
    uint64_t random_seed=0)
{
  /// Performs optimization with walksat and returns the tuple (final_state,
  /// num_steps, num_unsat)
  auto t0_ = std::chrono::high_resolution_clock::now();
  pysa::sat::Formula_WS formula_ws;
  for(const auto& cl: formula){
      pysa::sat::Clause_WS  _cl;
      _cl._lits.clear();
      for( auto l : cl)
        _cl.add_lit(pysa::sat::Lit_WS(std::abs(l) - 1, l<0));
      formula_ws.add_clause(std::move(_cl));
  }

  WalksatResult result = walksat_optimize_core(formula_ws, max_steps, p, max_unsat, random_seed);
  auto et_ = std::chrono::high_resolution_clock::now();
  result.preproc_time_us = std::chrono::duration_cast<std::chrono::microseconds>(et_ - t0_)
      .count() - result.computation_time_us;

  return result;
}

WalksatResult walksat_optimize_cnf_py(
    const std::string& cnf_file,
    uint64_t max_steps,
    double p=0.5,
    uint64_t max_unsat=0,
    uint64_t random_seed=0)
{
  /// Performs optimization with walksat and returns the tuple (final_state,
  /// num_steps, num_unsat)
  auto t0_ = std::chrono::high_resolution_clock::now();
  const auto formula_ws = [&cnf_file]() {
    if (auto ifs = std::ifstream(cnf_file); ifs.good())
      return pysa::algstd::ReadCNF(ifs);
    else
      throw std::runtime_error("Cannot open file: '" + cnf_file + "'");
  }();

  WalksatResult result = walksat_optimize_core(formula_ws, max_steps, p, max_unsat, random_seed);
  auto et_ = std::chrono::high_resolution_clock::now();
  result.preproc_time_us = std::chrono::duration_cast<std::chrono::microseconds>(et_ - t0_)
                               .count() - result.computation_time_us;

  return result;
}

PYBIND11_MODULE(bindings, m){
  py::class_<WalksatResultState>(m, "WalksatResultState",  py::buffer_protocol())
      .def_buffer([](WalksatResultState& rs){
        return py::buffer_info{
            rs.state.data(),
            sizeof(uint8_t),
            py::format_descriptor<uint8_t>::format(),
            1,
            {rs.state.size()},
            {1}
        };
      })
      ;
  py::class_<WalksatResult>(m, "_CDCLResult")
      .def_readonly("result_state", &WalksatResult::result_state)
      .def_readonly("num_unsat", &WalksatResult::num_unsat)
      .def_readonly("preproc_time_us", &WalksatResult::preproc_time_us)
      .def_readonly("computation_time_us", &WalksatResult::computation_time_us)
      .def_readonly("iterations", &WalksatResult::iterations)
      ;
  m.def("walksat_optimize",
        walksat_optimize_py
  );
  m.def("walksat_optimize_cnf",
        walksat_optimize_cnf_py
  );
}