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

from pysa_stern.bindings.libmld import _MLDProblem
from pysa_stern.bindings.libmld import _BitVec_u8, _BitVec_u16, _BitVec_u32, _BitVec_u64
from typing import TextIO


class BitVec:
    """
    Bit-packed vector
    """
    def __init__(self, n, dtype=np.uint8):
        if dtype == np.uint8:
            self._bit_vec = _BitVec_u8(n)
        elif dtype == np.uint16:
            self._bit_vec = _BitVec_u16(n)
        elif dtype == np.uint32:
            self._bit_vec = _BitVec_u32(n)
        elif dtype == np.uint64:
            self._bit_vec = _BitVec_u64(n)
        else:
            raise RuntimeError(f"Invalid dtype {dtype}")
        self._arr = np.asarray(self._bit_vec)

    def c_type(self):
        """
        Return the python binding of the BitVec C++ class
        :return:
        """
        return self._bit_vec

    @property
    def array(self):
        """
        Return a numpy array wrapping the underlying buffer
        :return:
        """
        return self._arr


class MLDProblem(_MLDProblem):
        
    def read_problem(self, f: TextIO):
        # Read in the entire file as a string
        prob_str = f.read()
        return self.read_problem_str(prob_str)


