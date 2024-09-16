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
#include <chrono>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cdcl.h"
namespace py = pybind11;

struct CDCLResultState{
  std::vector<uint8_t> state;
};

struct CDCLResult{
  std::optional<CDCLResultState> result_state;
  size_t preproc_time_us = 0;
  size_t computation_time_us = 0;
};

CDCLResult cdcl_optimize(const std::vector<std::vector<int32_t>>& fv, bool uip){

  auto t0_ = std::chrono::high_resolution_clock::now();

  FormulaT formula;
  for(const std::vector<int32_t>& clv: fv ){
    ClauseT cl;
    for (int32_t l : clv){
      cl.add_lit(Lit::from_int(l));
    }
    formula.add_clause(std::move(cl));
  }
  CDCL cdcl(std::move(formula));
  cdcl._uip = uip;

  auto it_ = std::chrono::high_resolution_clock::now();
  int rc = cdcl.run();
  auto et_ = std::chrono::high_resolution_clock::now();
  CDCLResult result;
  if(rc == CDCLSAT){
    result.result_state = CDCLResultState{cdcl.prop._state};
  }
  result.preproc_time_us = std::chrono::duration_cast<std::chrono::microseconds>(it_ - t0_)
      .count();
  result.computation_time_us = std::chrono::duration_cast<std::chrono::microseconds>(et_ - it_)
      .count();
  return result;
}
PYBIND11_MODULE(bindings, m){
  py::class_<CDCLResultState>(m, "CDCLResultState",  py::buffer_protocol())
    .def_buffer([](CDCLResultState& rs){
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
  py::class_<CDCLResult>(m, "_CDCLResult")
      .def_readonly("result_state", &CDCLResult::result_state)
      .def_readonly("preproc_time_us", &CDCLResult::preproc_time_us)
      .def_readonly("computation_time_us", &CDCLResult::computation_time_us)
  ;
  m.def("cdcl_optimize",
        cdcl_optimize
        );
}
