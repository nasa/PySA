# Author: Humberto Munoz Bauza (humberto.munozbauza@nasa.gov)
#
# Copyright © 2024, United States Government, as represented by the Administrator
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

import timeit
import numpy as np

from pysa2.base import Algorithm
from pysa2.models.sat import Satisfiability, SatDimacs, SatK, SATResult
from pysa2.algstd.sat import read_cnf
from pysa2_cdcl.bindings import cdcl_optimize, _CDCLResult


class CDCLResult(SATResult):
    def __init__(self, _cdcl_result: _CDCLResult):
        super().__init__()
        if _cdcl_result.result_state is not None:
            self._state = np.asarray(_cdcl_result.result_state, dtype=np.int8)
        else:
            self._state = None
        #self._computation_runtime = float(_cdcl_result.computation_time_us)*1e-6
        #self._total_runtime = self._computation_runtime + float(_cdcl_result.preproc_time_us)*1e-6

    def solution(self):
        return self._state

    def sat_configurations(self):
        if self._state is not None:
            yield self._state
        return


class CDCLOptimizer(Algorithm):

    def __init__(self, verbose=False) -> None:
        super().__init__()
        self.verbose = verbose

    def instance_class(self):
        return [SatDimacs, SatK]

    def solve(self, inst: Satisfiability):
        t0 = timeit.default_timer()
        if isinstance(inst, SatDimacs):
            formula = read_cnf(inst.dimacs_string()).as_list()
        elif isinstance(inst, SatK):
            formula = inst.clauses
        else:
            raise RuntimeError("Instance is not a valid type.")

        # Optimize
        t1 = timeit.default_timer()
        res = cdcl_optimize(formula)
        t2 = timeit.default_timer()
        result = CDCLResult(res)
        result._total_runtime = t2 - t0
        result._computation_runtime = t2 - t1
        return result
