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
from mceliece.tseytin.bindings import into_3sat_only, add_weight_constraints, xorsat_from_numpy_arrays

def reduce_mld_to_sat(G: np.ndarray, y: np.ndarray, t: int, as_3sat_only=False):
    """
    Generate a DIMACS string for the reduced MLD problem as either a 3-SAT + XORSAT problem
    or as a pure 3-SAT problem.
    The Tseytin reduction is used to enforce the Hamming weight t constraint on the error string
    as an accumulation circuit, which is then reduced to 3-SAT clauses.
    """
    xs = xorsat_from_numpy_arrays(G, y)
    n = xs.nvars()
    if as_3sat_only:
        into_3sat_only(xs)
    add_weight_constraints(t, xs, n)
    dimacs_str = xs.as_dimacs_str()
    return dimacs_str
