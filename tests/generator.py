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
from mceliece.generator import McElieceInstance, NiederInstance
from mceliece.goppa import generate_random_goppa
import logging

from mceliece.randutil import random_weight_t


class TestMcElieceInstances(unittest.TestCase):

    def test_gen(self):
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s %(levelname)s] %(message)s")
        n = 50
        rng = np.random.default_rng(1234)
        for t in range(2, 9):
            for i in range(100):
                goppa_code = generate_random_goppa(n, t, rng=rng, exact_k=True)
                instance = McElieceInstance(goppa_code, rng)
                pub, priv = instance.public_private_pair()
                m = rng.integers(0, 2, (1, goppa_code.k), dtype=np.int8)
                err_pos = rng.choice(n, t, replace=False)
                y = pub.encode(m)
                y2 = np.copy(y)
                y2[0, err_pos] = y2[0, err_pos] ^ 1
                m2, e2 = priv.decode(y2)
                self.assertTrue(np.all(m2 == m))

                err = np.reshape(random_weight_t(n, t, rng), (1, n))
                instance = NiederInstance(goppa_code, rng, t=t)
                pub, priv = instance.public_private_pair()
                y = pub.encode(err)
                err2 = priv.decode(y)
                self.assertTrue(np.all(err2 == err))
