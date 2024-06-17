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

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sternx/stern.h>

namespace py = pybind11;

// Wrapper class to make solution accessible as a numpy array
struct PySA_Stern_solution{
  std::vector<uint8_t> vec;
};

template<typename T>
void init_bitvec_h(py::module& m, const char* t_name){
  py::class_<BitVec<T>>(m, t_name, py::buffer_protocol())
          .def(py::init<uint64_t>())
          .def("set", &BitVec<T>::set, py::doc{"Set bit to bool value"})
          .def("flip", &BitVec<T>::flip, py::doc{"Flip bit value"})
          .def("clear", &BitVec<T>::clear, py::doc{"Reset bit to 0"})
          .def("num_blocks", &BitVec<T>::num_blocks, py::doc{"Number of blocks in the memory array"})
          .def("size", &BitVec<T>::size, py::doc{"Number of bits in the bit vector."} )
          .def("__repr__", [](const BitVec<T>& bv){
              std::stringstream oss;
              oss << bv;
              return oss.str();
          })
          .def_buffer([](BitVec<T>& bv){
            return py::buffer_info{
              bv.data_ptr(),
              sizeof(T),
              py::format_descriptor<T>::format(),
              1,
              {bv.num_blocks()},
              {1}
            };
          })
          .doc() = "Packed bit-vector array using blocks of types u8, u16, u32, or u64. "
                   "Underlying memory supports the numpy buffer protocol."
          ;
}

void init_mld_h(py::module& m){
  py::enum_<MLDType>(m, "MLDType")
          .value("MLD_G", MLDType::G)
          .value("MLD_H", MLDType::H)
          ;

  py::class_<MLDProblem>(m, "_MLDProblem")
          .def(py::init<>())
          .def("problem_type", &MLDProblem::problem_type)
          .def("NVars", &MLDProblem::NVars)
          .def("NClauses", &MLDProblem::NClauses)
          .def("is_parcheck", &MLDProblem::is_parcheck)
          .def("CodeLength", &MLDProblem::CodeLength)
          .def("CodeDim", &MLDProblem::CodeDim)
          .def("Weight", &MLDProblem::Weight)
          .def("clause_list", &MLDProblem::clause_list)
          .def("read_problem_str", &MLDProblem::read_problem_string)
          .doc() =  "Options definding a MLD problem."
          ;
}

void init_libmld_submod(py::module& m){
  auto _docstr =  R"pbdoc(
PySA-libmld: Utility module for defining maximum likelihood decoding problems.
)pbdoc";
  auto submod = m.def_submodule("libmld", _docstr);
  init_bitvec_h<uint8_t>(submod, "_BitVec_u8");
  init_bitvec_h<uint16_t>(submod, "_BitVec_u16");
  init_bitvec_h<uint32_t>(submod, "_BitVec_u32");
  init_bitvec_h<uint64_t>(submod, "_BitVec_u64");
  init_mld_h(submod);
}

PYBIND11_MODULE(bindings, m){
    m.doc() = R"pbdoc(
PySA-sternx: Stern algorithm for unstructured decoding problems.
)pbdoc";

    py::class_<sternc_opts>(m, "SternOpts")
        .def(py::init<>())
        .def_readwrite("parcheck", &sternc_opts::parcheck)
        .def_readwrite("bench", &sternc_opts::bench)
        .def_readwrite("test_hw1", &sternc_opts::test_hw1)
        .def_readwrite("t", &sternc_opts::t)
        .def_readwrite("max_iters", &sternc_opts::max_iters)
        .def_readwrite("l", &sternc_opts::l)
        .def_readwrite("p", &sternc_opts::p)
        .def_readwrite("m", &sternc_opts::m)
        .doc() =  "Options for the Stern algorithm routine."
        //.def_readwrite("nvars", &sternc_opts::nvars)
    ;
    py::class_<PySA_Stern_solution>(m, "PySASternSolution", py::buffer_protocol())
       .def_buffer([](PySA_Stern_solution& sol){
            return py::buffer_info{
              sol.vec.data(),
              1,
              py::format_descriptor<uint8_t>::format(),
              1,
              {sol.vec.size()},
              {1}
            };
          })
    ;
    m.def("sterncpp_adjust_opts", &sterncpp_adjust_opts);
    m.def("sterncpp_main", &sterncpp_main);
  init_libmld_submod(m);
}