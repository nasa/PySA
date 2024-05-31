/*
Author: Salvatore Mandra (salvatore.mandra@nasa.gov)

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

#include <pysa/dpll/dpll.hpp>
#include <pysa/sat/sat.hpp>
#include <sstream>

using namespace pysa::dpll;
namespace py = pybind11;

#ifdef USE_MPI
#include <mpi.h>

inline MPI_Comm mpi_comm_world;
inline int mpi_rank;
inline int mpi_size;

auto bcast_cnf(const py::object cnf, const std::size_t root) {
  using cnf_type_ = std::vector<std::vector<int32_t>>;
  return py::cast(pysa::dpll::mpi::_Bcast(
      mpi_comm_world,
      cnf && !cnf.is(py::none()) ? cnf.cast<cnf_type_>() : cnf_type_{}, root));
}

// Initialize MPI
void init_MPI(py::module m) {
  m.def(
      "init_MPI",
      []() {
        int provided;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        if (MPI_THREAD_MULTIPLE != provided)
          throw std::runtime_error("Cannot initialize MPI.");
      },
      "Initialize the MPI environment.");

  m.def(
      "finalize_MPI", []() { MPI_Finalize(); },
      "Finalize the MPI environment.");

  m.def(
      "setup_MPI",
      []() {
        // Check thread level
        {
          int thread_level_;
          MPI_Query_thread(&thread_level_);
          if (thread_level_ != MPI_THREAD_MULTIPLE)
            throw std::runtime_error(
                "MPI should be set to 'MPI_THREAD_MULTIPLE'");
        }

        // Duplicate worlds
        MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_world);

        // Get MPI rank and size
        MPI_Comm_rank(mpi_comm_world, &mpi_rank);
        MPI_Comm_size(mpi_comm_world, &mpi_size);
      },
      "Setup the MPI environment.");

  m.def("bcast_cnf", &bcast_cnf, py::arg("cnf"), py::pos_only(),
        py::arg("root"), "Broadcast CNF using MPI.");

  m.def(
      "get_rank", []() { return mpi_rank; }, "Get MPI rank.");

  m.def(
      "get_size", []() { return mpi_size; }, "Get MPI size.");
}
#endif

void init_GetRandomInstance(py::module m) {
  static constexpr auto __doc__ =
      R"(Generate random k-SAT instances.

Parameter
---------
k: int
  Number of variables in each clause.
n_variables: int
  Number of variables to use.
n_clauses: int
  Number of clauses to use.
seed: int, optional
  Seed to use.
)";
  m.def(
      "get_random_instance",
      [](std::size_t n, std::size_t m, std::size_t k,
         std::optional<std::size_t> seed) {
        return sat::GetRandomInstance(k, n, m, seed);
      },
      py::arg("n_variables"), py::arg("n_clauses"), py::pos_only(),
      py::arg("k") = 3, py::kw_only(), py::arg("seed") = py::none(), __doc__);
}

void init_ReadCNF(py::module m) {
  static constexpr auto __doc__ =
      R"(Read CNF from string.

Parameter
---------
cnf: str
  String to use to read CNF.
)";
  m.def(
      "loads",
      [](const std::string &cnf) {
        std::istringstream ss_{cnf};
        return sat::ReadCNF(ss_);
      },
      py::arg("cnf"), py::pos_only(), __doc__);
}

void init_BitSet(py::module m) {
  using self_type = pysa::dpll::BitSet<std::size_t, std::vector>;
  py::class_<self_type>(m, "BitSet")
      .def("__getitem__", &self_type::test)
      .def("tolist",
           [](const self_type &self) {
             std::vector<bool> list_(std::size(self));
             for (std::size_t i_ = 0; i_ < std::size(self); ++i_)
               list_[i_] = self.test(i_);
             return list_;
           })
      .def("__repr__", [](const self_type &self) { return std::string(self); });
}

void init_Branch(py::module m) {
  using self_type = pysa::dpll::sat::Branch<
      pysa::dpll::BitSet<std::size_t, std::vector>, std::list, std::vector,
      pysa::dpll::sat::Instance<pysa::dpll::BitSet<std::size_t, std::vector>,
                                std::vector>>;
  py::class_<self_type>(m, "Branch")
      .def_property_readonly("state", &self_type::state)
      .def_property_readonly("partial_sat", &self_type::partial_sat)
      .def_property_readonly("pos", &self_type::pos)
      .def_property_readonly("n_sat", &self_type::n_sat)
      .def_property_readonly("n_unsat", &self_type::n_unsat)
      .def("__repr__", [](const self_type &self) {
        return "Branch(" + std::string(self.state()) +
               ", pos=" + std::to_string(self.pos()) +
               ", n_sat=" + std::to_string(self.n_sat()) +
               ", n_unsat=" + std::to_string(self.n_unsat()) + ")";
      });
}

void init_Configuration(py::module m) {
  using self_type =
      sat::Configuration<pysa::dpll::BitSet<std::size_t, std::vector>,
                         std::size_t>;
  py::class_<self_type>(m, "Configuration")
      .def_property_readonly("state",
                             [](const self_type &self) { return self.state; })
      .def_property_readonly("n_unsat",
                             [](const self_type &self) { return self.n_unsat; })
      .def("__repr__", [](const self_type &self) {
        return "Configuration(" + std::string(self.state) +
               ", n_unsat=" + std::to_string(self.n_unsat) + ")";
      });
}

void init_Optimize(py::module m) {
  static constexpr auto __doc__ =
      R"(Optimize CNF formula.

Parameter
---------
cnf: list[list[int]]
  CNF to optimize.
max_n_unsat: int, optional
  Number of maximum allowed unsatisfied clauses.
n_threads: int, optional
  Number of threads to use (by default, all cores will be used)
walltime: float, optional
  Maximum number of seconds to run the optimization.
verbose: bool, optional
  Verbose output.
)";
  m.def(
      "optimize",
      [](const std::vector<std::vector<int32_t>> cnf,
         const std::size_t max_n_unsat,
         const std::optional<std::size_t> n_threads,
         const std::optional<float> walltime, const bool verbose) {
#ifdef USE_MPI
        if (walltime)
          throw std::logic_error(
              "At the moment, 'walltime' is not implemented for "
              "MPI.");
        else
          return sat::mpi::optimize(mpi_comm_world, cnf, max_n_unsat, verbose,
                                    n_threads);
#else
        if (walltime)
          return sat::optimize(
              cnf, max_n_unsat, verbose, n_threads,
              std::chrono::milliseconds(
                  static_cast<std::size_t>(walltime.value() * 1e3)));
        else
          return sat::optimize(cnf, max_n_unsat, verbose, n_threads, nullptr);
#endif
      },
      py::arg("cnf"), py::pos_only(), py::arg("max_n_unsat") = 0, py::kw_only(),
      py::arg("n_threads") = py::none(),
      py::arg("walltime") = std::numeric_limits<float>::infinity(),
      py::arg("verbose") = false, __doc__);
}

// Documentation
static constexpr auto __doc__ = R"(DPLL-SAT.)";

// Initialize main module
PYBIND11_MODULE(pysa_dpll_core, m) {
  m.doc() = __doc__;
#ifdef USE_MPI
  init_MPI(m);
#endif

  // Initialze SAT module
  py::module sat_m_ = m.def_submodule("sat", "SAT");
  init_BitSet(sat_m_);
  init_Branch(sat_m_);
  init_Configuration(sat_m_);
  init_Optimize(sat_m_);

  // Initialize utilities for SAT
  py::module sat_utils_m_ = sat_m_.def_submodule("utils", "Common utilities");
  init_GetRandomInstance(sat_utils_m_);
  init_ReadCNF(sat_utils_m_);
}
