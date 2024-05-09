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
from mceliece.f2mat import f2_matmul
from mceliece.gausselm import gauss_jordan_unpacked
from mceliece.lincodes import LinearCode
from mceliece.goppa import GoppaCode, generate_random_goppa
from mceliece.randutil import random_invertible_f2, random_perm_f2


class McEliecePublicKey:

    def __init__(self, Gp):
        self.Gp = Gp

    def encode(self, m):
        return f2_matmul(m, self.Gp)


class McEliecePrivateKey:

    def __init__(self, Sinv, P, code: LinearCode):
        self.Sinv = Sinv
        self.P = P
        self.code = code

    def decode(self, y):
        """
        Decode the message using the known structure of the public key Gp = S G P
        :param y:
        :return: The tuple (m, e) of the encoded message and applied error
        """
        y2 = f2_matmul(y, np.transpose(self.P))
        y3, e = self.code.decode(y2)
        m = f2_matmul(y3, self.Sinv)
        return m, e


class NiederPublicKey:
    """
    Niederreiter Public Key
    """

    def __init__(self, Hp, t=None):
        self.Hp = Hp
        self.t = t

    def encode(self, m):
        if self.t is not None:
            w = np.count_nonzero(m, axis=-1)
            if not np.all(w <= self.t):
                raise ValueError(
                    f"(Niederreiter encoding) Hamming weight of message is larger that {self.t}"
                )
        return f2_matmul(m, np.transpose(self.Hp))


class NiederPrivateKey(McEliecePrivateKey):

    def __init__(self, Sinv, P, code: LinearCode):
        super().__init__(Sinv, P, code)
        n, k = self.code.n, self.code.k
        Htaug = np.concatenate(
            [self.code.H.transpose(),
             np.eye(n, dtype=np.int8)], axis=1)

        Htgj, r, _ = gauss_jordan_unpacked(Htaug, upto=n - k)
        assert r == n - k
        self.Hpinv = Htgj[:n - k, n - k:]  # [n-k, n]
        self.decmat = f2_matmul(Sinv.transpose(), self.Hpinv)

    def decode(self, c):
        """
        Decode the message using the known structure of the public key Hp = S H P
        Simply reverses the tuple returned by McEliecePrivateKey
        """
        # find z such that  H z^T = S^-1 c
        z = f2_matmul(c, self.decmat)
        m, e = self.code.correct_err(z)
        err = f2_matmul(e, self.P)
        return err


class McElieceInstance:

    def __init__(self, code: LinearCode, rng: np.random.Generator = None):
        self.code = code
        self.n = code.n
        if rng is None:
            rng = np.random.default_rng()
        s, sinv = random_invertible_f2(self.code.k, rng)
        p = random_perm_f2(self.n, rng)
        self.smat = s
        self.sinv = sinv
        self.p = p
        self.Gp = f2_matmul(s, f2_matmul(code.G, p))

    def public_private_pair(self):
        pub = McEliecePublicKey(self.Gp)
        priv = McEliecePrivateKey(self.sinv, self.p, self.code)
        return pub, priv


class NiederInstance:

    def __init__(self,
                 code: LinearCode,
                 rng: np.random.Generator = None,
                 t=None):
        self.code = code
        self.n = code.n
        if rng is None:
            rng = np.random.default_rng()
        s, sinv = random_invertible_f2(self.n - self.code.k, rng)
        p = random_perm_f2(self.n, rng)
        self.smat = s
        self.sinv = sinv
        self.p = p
        self.Hp = f2_matmul(s, f2_matmul(code.H, p))
        self.t = None

    def public_private_pair(self):
        pub = NiederPublicKey(self.Hp, t=self.t)
        priv = NiederPrivateKey(self.sinv, self.p, self.code)
        return pub, priv
