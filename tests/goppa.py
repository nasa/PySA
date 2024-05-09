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

import unittest
import numpy as np
from copy import copy
from mceliece.goppa import generate_random_goppa, GoppaCode
import galois
import logging


class TestGoppa(unittest.TestCase):

    def test_goppa_tm1(self):
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s %(levelname)s] %(message)s")
        rng = np.random.default_rng(1234)
        n = 50

        for t in range(3, 9):
            for i in range(20):
                goppa_code = generate_random_goppa(n, t, rng=rng, exact_k=True)
                m = rng.integers(0, 2, (1, goppa_code.k), dtype=np.int8)
                err_pos = rng.choice(n, t - 1, replace=False)
                y = goppa_code.encode(m)
                self.assertTrue(goppa_code.is_codeword(y[0]))
                y2 = np.copy(y)
                y2[0, err_pos] = y2[0, err_pos] ^ 1
                m2, e2 = goppa_code.decode(y2)
                matched = np.all(m2 == m)
                self.assertTrue(matched)

    def test_goppa_t(self):
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s %(levelname)s] %(message)s")
        rng = np.random.default_rng(1234)
        n = 50
        for t in range(2, 7):
            for dt in range(1, t + 1):
                for i in range(20):
                    goppa_code = generate_random_goppa(n,
                                                       t,
                                                       rng=rng,
                                                       exact_k=True)
                    msg = rng.integers(0, 2, (1, goppa_code.k), dtype=np.int8)
                    err_pos = rng.choice(n, dt, replace=False)
                    y = goppa_code.encode(msg)
                    self.assertTrue(goppa_code.is_codeword(y[0]))
                    y2 = np.copy(y)
                    y2[0, err_pos] = y2[0, err_pos] ^ 1
                    m2, e2 = goppa_code.decode(y2)
                    matched = np.all(m2 == msg)
                    self.assertTrue(matched)
