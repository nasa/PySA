# Author: Salvatore Mandra (salvatore.mandra@nasa.gov)
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
from __future__ import annotations
from . import rc

__all__ = ['MPI']

try:
    from pysa_dpll_core import (init_MPI, finalize_MPI, setup_MPI, get_rank,
                                get_size, bcast_cnf)

    class MPI_env:

        def __init__(self):
            if rc.initialize:
                init_MPI()
            setup_MPI()

        def __del__(self):
            if rc.finalize:
                finalize_MPI()

        @property
        def rank(self):
            return get_rank()

        @property
        def size(self):
            return get_size()

        @property
        def enabled(self):
            return True

        def bcast_cnf(self, cnf: list[list[int]], root: int):
            return bcast_cnf(cnf, root)

except ImportError:

    class MPI_env:

        @property
        def rank(self):
            return 0

        @property
        def size(self):
            return 1

        @property
        def enabled(self):
            return False


# Start environment
MPI = MPI_env()
