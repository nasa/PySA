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
from io import StringIO
from itertools import combinations
from math import comb
from nistrng import check_eligibility_all_battery, run_all_battery, SP800_22R1A_BATTERY
import numpy as np

from mceliece.f2mat import f2_matmul, f2_sum_to, invert_triangular, F2Matrix
from mceliece.randutil import random_perm_f2, random_invertible_f2
from mceliece.lincodes import enumerate_syndromes, hw_table


class TestF2Mat(unittest.TestCase):

    def test_basic(self):
        a = np.asarray([[1, 1], [0, 1]], dtype=np.int8)
        b = np.asarray([[1, 0], [1, 1]], dtype=np.int8)
        c = np.asarray([[0, 1], [1, 0]], dtype=np.int8)
        out = np.zeros((2, 2), dtype=np.int8)
        f2_sum_to([a, b], out)
        self.assertTrue(np.all(np.equal(c, out)))

        d = np.asarray([[0, 1], [1, 1]])
        d2 = f2_matmul(a, b)
        self.assertTrue(np.all(np.equal(d, d2)))

    def test_inv_tri(self):
        a = np.eye(8, dtype=np.int8)
        ainv = invert_triangular(a)
        c = f2_matmul(a, ainv)
        self.assertTrue(np.all(np.equal(c, np.eye(8, dtype=np.int8))))

        a = np.asarray([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]])
        ainv = invert_triangular(a)
        c = f2_matmul(a, ainv)
        self.assertTrue(np.all(np.equal(c, np.eye(4, dtype=np.int8))))

    def test_random_mats(self):
        rng = np.random.default_rng(1234)
        p = random_perm_f2(8, rng)
        b = f2_matmul(p.transpose(), p)
        self.assertTrue(np.all(np.equal(b, np.eye(8, dtype=np.int8))))

        r, rinv = random_invertible_f2(8, rng)
        c = f2_matmul(r, rinv)
        self.assertTrue(np.all(np.equal(c, np.eye(8, dtype=np.int8))))

        nn = 2**10
        r, rinv = random_invertible_f2(nn, rng)
        c = f2_matmul(r, rinv)
        self.assertTrue(np.all(np.equal(c, np.eye(nn, dtype=np.int8))))

    def test_codes(self):
        rng = np.random.Generator(np.random.PCG64DXSM(1659269624))
        #rng = np.random.default_rng(1659269624)
        # For this test:
        n = 32  # code size
        k = 8  # message size
        # We simply get the BKLC generator and check matrices from Magma

        bklc_32_8_string = \
 """
 1 0 0 1 0 0 0 0 0 1 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 
 0 1 0 1 0 0 0 0 0 0 1 1 0 1 1 1 0 1 0 1 0 0 0 0 0 0 1 1 1 0 1 1 
 0 0 1 1 0 0 0 0 0 1 1 0 0 1 1 1 1 1 0 0 1 1 1 1 1 0 0 1 1 1 1 1 
 0 0 0 0 1 0 0 0 0 1 1 1 0 0 0 1 0 1 1 0 0 1 0 0 1 1 0 1 1 1 0 1 
 0 0 0 0 0 1 0 0 0 0 1 0 1 0 1 0 1 0 0 1 0 1 1 1 0 1 1 1 0 0 1 0 
 0 0 0 0 0 0 1 0 0 0 0 1 1 1 1 0 0 1 1 0 1 1 1 0 1 0 1 1 1 1 1 0 
 0 0 0 0 0 0 0 1 0 1 0 0 1 0 1 0 0 1 1 0 1 1 0 1 1 1 1 0 0 0 0 1 
 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 
 """
        bklc_32_8_dual_string = \
"""
 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 0 1 1 0 0 
 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 
 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 1 
 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0 1 
 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 1 1 1 1 0 
 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 1 0 1 0 0 
 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1 1 1 1 
 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 
 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 1 
 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 
 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 
 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1 1 0 1 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 1 0 1 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 1 0 1 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 1 1 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 1 1 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 1 1 0 1 1 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0 1 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0 1 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 1 1 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 1 
"""

        bklc_32_8 = np.loadtxt(StringIO(bklc_32_8_string),
                               dtype=np.int8)  # [8, 32]
        bklc_32_8_dual = np.loadtxt(StringIO(bklc_32_8_dual_string),
                                    dtype=np.int8)  # [24, 32]
        m = bklc_32_8_dual.shape[0]  # dual size
        c = f2_matmul(bklc_32_8, bklc_32_8_dual.transpose())
        self.assertTrue(np.all(c == np.zeros((k, m), dtype=np.int8)))
        # this code is a d=13 code, so we enumerate the syndromes of errors up to weight 6
        # For speed we will restrict to weight 4 errors
        synd_dict = enumerate_syndromes(bklc_32_8_dual, 4)
        # Generate a random invertible matrix
        r, rinv = random_invertible_f2(n, rng)
        c = f2_matmul(r, rinv)
        self.assertTrue(np.all(c == np.eye(n, dtype=np.int8)))

        # Generate a random 8-bit sequence
        msglen = 2**10
        rand_msg_arr = rng.integers(0, 256, (msglen, 1), dtype=np.uint8)
        rand_msg = F2Matrix(rand_msg_arr)
        Gpacked = F2Matrix(bklc_32_8.transpose())
        Hpacked = F2Matrix(bklc_32_8_dual)
        encoded = rand_msg @ Gpacked  # [msglen, 32]
        orig_msg = encoded.copy()
        hw4tbl = hw_table(n, 4)
        # Randomly apply weight 4 errors
        rand_errors = rng.choice(hw4tbl, size=msglen,
                                 axis=0)  # [msglen, 4] ints
        for i, idx in enumerate(rand_errors):
            for j in idx:
                encoded[i, j] = (encoded[i, j] + 1) % 2
        packed_encoded = F2Matrix(encoded)
        flat_encoded = encoded.flatten()
        # Check the eligibility of the test and generate an eligible battery from the default NIST-sp800-22r1a battery
        eligible_battery: dict = check_eligibility_all_battery(
            flat_encoded, SP800_22R1A_BATTERY)
        # Print the eligible tests
        print("Eligible test from NIST-SP800-22r1a:")
        for name in eligible_battery.keys():
            print("-" + name)
        # Test the sequence on the eligible tests
        results = run_all_battery(flat_encoded, eligible_battery, False)
        # Print results one by one
        print("Test results:")
        for result, elapsed_time in results:
            if result.passed:
                print("- PASSED - score: " + str(np.round(result.score, 3)) +
                      " - " + result.name + " - elapsed time: " +
                      str(elapsed_time) + " ms")
            else:
                print("- FAILED - score: " + str(np.round(result.score, 3)) +
                      " - " + result.name + " - elapsed time: " +
                      str(elapsed_time) + " ms")

        # Perform error correction
        syndromes = packed_encoded @ Hpacked  # [msglen, 24]
        errs = [synd_dict[tuple(np.packbits(s))] for s in syndromes]
        corrected_msg = (encoded + errs) % 2
        self.assertTrue(np.all(corrected_msg == orig_msg))
