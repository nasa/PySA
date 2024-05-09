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
import logging
import tempfile

from mceliece.genmci import genmci


class TestGenMCI(unittest.TestCase):

    def test_genmci_50(self):
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s %(levelname)s] %(message)s")

        n = 50
        rng = np.random.default_rng(1234)
        for t in range(2, 7):
            with tempfile.TemporaryDirectory() as dir_name:
                genmci(n,
                       t,
                       100,
                       dir_name+"/_test50_dir_pub",
                       dir_name+"/_test50_dir_priv",
                       seed=rng.integers(2**31))

    def test_genmci_50_nied(self):
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s %(levelname)s] %(message)s")

        n = 50
        rng = np.random.default_rng(1234)
        for t in range(2, 7):
            with tempfile.TemporaryDirectory() as dir_name:
                genmci(n,
                       t,
                       100,
                       dir_name+"/_test50_dir_pub",
                       dir_name+"/_test50_dir_priv",
                       nieder=True,
                       seed=rng.integers(2 ** 31))

    def test_genmci_100(self):
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s %(levelname)s] %(message)s")
        n = 100
        rng = np.random.default_rng(1234)
        for t in range(10, 11, 12):
            with tempfile.TemporaryDirectory() as dir_name:
                genmci(n,
                       t,
                       10,
                       dir_name+"/_test100_dir_pub",
                       dir_name+"/_test100_dir_priv",
                       f2_deg=8,
                       seed=rng.integers(2 ** 31))

    def test_genmci_bad_neg(self):
        with self.assertRaises(RuntimeError):
            with tempfile.TemporaryDirectory() as dir_name:
                genmci(-5, 2, 1, dir_name+"/_t1", dir_name+"/_t2", seed=1234)

    def test_genmci_bad_cond1(self):
        with self.assertRaises(RuntimeError):
            with tempfile.TemporaryDirectory() as dir_name:
                genmci(40, 11, 1, dir_name+"/_t1", dir_name+"/_t2", f2_deg=6, seed=1234)  # violates k = n -tm > 0

    def test_genmci_bad_empty(self):
        with self.assertRaises(RuntimeError):
            with tempfile.TemporaryDirectory() as dir_name:
                genmci(10, 2, 100, dir_name+"/_t1", dir_name+"/_t2", seed=1234, overwrite=True)

    def test_overwrite(self):
        with self.assertRaises(FileExistsError):
            with tempfile.TemporaryDirectory() as dir_name:
                genmci(48, 6, 1, dir_name + "/_t1", dir_name + "/_t2", f2_deg=6, seed=1234)
                genmci(48, 6, 1, dir_name + "/_t1", dir_name + "/_t2", f2_deg=6, seed=1234)


