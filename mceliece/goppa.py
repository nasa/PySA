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
from mceliece.gausselm import standardize_parcheck, gauss_jordan_unpacked, null_space
import galois
from galois import Poly
from mceliece.lincodes import LinearCode
import logging
from typing import Type, List
from tqdm.auto import tqdm


def list_egcd(a: Poly, b: Poly):
    """
        Modification of galois.egcd that returns all iterations of the extended Euclidean algorithm
    """
    if not a.field is b.field:
        raise ValueError(
            f"Polynomials `a` and `b` must be over the same Galois field, not {a.field} and {b.field}."
        )

    field = a.field
    zero = Poly([0], field=field)
    one = Poly([1], field=field)

    r = [a, b]
    s = [one, zero]
    t = [zero, one]
    i = 1
    while r[i] != 0:
        q = r[i - 1] // r[i]
        r.append(r[i - 1] - q * r[i])
        s.append(s[i - 1] - q * s[i])
        t.append(t[i - 1] - q * t[i])
        i += 1

    # Make the GCD polynomial monic
    for i, ri in enumerate(r):
        c = ri.coeffs[0]  # The leading coefficient
        if c > 1:
            r[i] = r[i] // c
            s[i] = s[i] // c
            t[i] = t[i] // c

    return r, s, t


def _poly_random_search(order: int,
                        degree: int,
                        test: str,
                        rng=None) -> galois.Poly:
    """
    Searches for a random monic polynomial of specified degree, only yielding those that pass the specified test
    (either 'is_irreducible()' or 'is_primitive()').
    """
    if rng is None:
        rng = np.random.default_rng()
    assert test in ["is_irreducible", "is_primitive"]
    field = galois.GF(order)

    # Only search monic polynomials of degree m over GF(q)
    while True:
        #integer = rng.integers(start, stop - 1, dtype=int)
        poly = galois.Poly.Random(degree, seed=rng, field=field)
        if getattr(poly, test)():
            return poly


class GoppaCode(LinearCode):

    def __init__(self, gf: Type[galois.FieldArray], code, g_poly: galois.Poly):
        self.gf = gf
        self.m = gf.degree
        self._n = len(code)
        self.code = code
        self.code_pos = {int(a): i for i, a in enumerate(code)}
        self.g_poly = g_poly
        self.g_coefs = g_poly.coefficients()
        self.t = g_poly.degree
        self.a = gf.primitive_element
        self.irred_poly = gf.irreducible_poly

        t = self.t
        n = self._n
        m = self.m
        # Construct the parity check matrix
        Cmat = np.stack([
            np.concatenate([self.g_coefs[i::-1],
                            gf.Zeros(t - i - 1)]) for i in range(0, t)
        ])
        #Xmat = np.stack([code ** i for i in range(t)])

        g_eval = g_poly(code)
        assert g_eval.min() > 0
        Xmat = (code[np.newaxis, :]**(np.arange(t, dtype=int))[:, np.newaxis])
        g_inv = g_eval**-1
        XYmat = Xmat * g_inv[np.newaxis, :]
        Hmat = Cmat @ XYmat
        Hv = gf.vector(Hmat)
        Hv = Hv.transpose((1, 0, 2)).copy()
        Hexpand = np.reshape(Hv, (n, t * self.m)).astype(np.int8, subok=False)
        Ht = np.copy(Hexpand)
        H = np.copy(Ht.transpose())
        self._H = H  #[tm, n]
        self._Ht = Ht
        gf2 = galois.GF(2)
        HGF2 = gf2(H)
        _GtGF2 = HGF2.null_space()
        #self._Gt = null_space(H)  # [n, k]
        self._G = _GtGF2.astype(np.int8, subok=False)  # [k, n]
        self._Gt = self._G.transpose().copy()
        k = self._G.shape[0]
        assert k >= n - m * t
        self._k = k
        self._nmk = self.n - self.k
        # Estimate code distance
        hweights = np.sum(self._G, axis=1)
        dists = np.sum(np.bitwise_xor(self._G[:, np.newaxis, :],
                                      self._G[np.newaxis, :, :]),
                       axis=2)
        d_arr = np.concatenate([hweights, dists.flatten()])
        ds = np.unique(d_arr)
        self._d = ds[1]
        # evaluate decoding matrix
        Gtaug = np.concatenate([self._Gt, np.eye(n, dtype=np.int8)], axis=1)

        Gtgj, r, _ = gauss_jordan_unpacked(Gtaug, upto=k)
        assert r == k
        self._decmat = Gtgj[:k, k:]

        # Evaluate the inverse of the polynomial (z-\alpha) for each code element \alpha
        # in the quotient ring GF(2^m)[z] / <g(z)>
        # 1.17 of Singh (2020)
        #
        # The factors of the sigma polynomial \sigma(z) = prod_i (z-\alpha_i)
        sigma_factors = [galois.Poly([1, -a], gf) for a in self.code]
        sigma_inv_factors = []
        for ga, gai, sf in zip(g_eval, g_inv, sigma_factors):
            g2 = g_poly - ga  # alpha is a root of this polynomial
            _p = g2 // sf
            _sfinv = (-_p * gai) % g_poly
            _prd = _sfinv * sf
            _prdmod = _prd % g_poly
            assert _prdmod == galois.Poly.One(gf)
            sigma_inv_factors.append(_sfinv)

        self.sigma_factors = sigma_factors
        self.sigma_inv_factors = sigma_inv_factors

    @property
    def G(self):
        return self._G

    @property
    def H(self):
        return self._H

    @property
    def n(self):
        return self._n

    @property
    def k(self):
        return self._k

    @property
    def nmk(self):
        return self._nmk

    def _id(self):
        return galois.Poly.Identity(self.gf)

    def _one(self):
        return galois.Poly.One(self.gf)

    def _zero(self):
        return galois.Poly.Zero(self.gf)

    def syndrome_polynomial(self, yvec: np.ndarray):
        """
        Compute the syndrome polynomial
            S_y(z) = \sum_{i=1}^n  y_i/(z-\alpha_i).
        The syndrome polynomial is zero modulo g(z) for an error-free codeword
        :param yvec:
        :return:
        """
        Sy = galois.Poly.Zero(self.gf)
        for i, si in enumerate(self.sigma_inv_factors):
            yi = self.gf(yvec[i])
            Sy = Sy + yi * si
        Sy = Sy % self.g_poly
        return Sy

    def is_codeword(self, yvec: np.ndarray):
        Sy = self.syndrome_polynomial(yvec)
        return Sy == self._zero()

    def encode(self, m: np.ndarray):
        return f2_matmul(m, self.G)

    def correct_err(self, y: np.ndarray):
        """
        Correct errors using Patterson's algorithm (see Patterson (1975) IEEE Trans. Info. Theory)

        :param y:
        :return:
        """
        y2 = np.copy(y)
        errs = np.zeros_like(y)
        for yj, ej in zip(y2, errs):
            Sy = self.syndrome_polynomial(yj)

            if Sy != galois.Poly.Zero(self.gf):  # correct errors
                # Invert Sy
                gcp, Syinv, _ = galois.egcd(Sy, self.g_poly)
                assert gcp == galois.Poly.One(self.gf)
                Pz: galois.Poly = Syinv + galois.Poly.Identity(self.gf)
                if Pz != galois.Poly.Zero(self.gf):
                    # compute the square root of P(z).
                    # In GF(2^m), sqrt(c) = c ** (2**(m-1)).
                    # Since GF(2^m)[z] / <g(z)> is isomorphic to GF(2^(mt)),
                    #   sqrt[P] = P ** (2**(mt -1))
                    # However, we need to be clever about this for large mt
                    sqrtPz = Pz
                    for _ in range(self.m * self.t - 1):
                        sqrtPz = (sqrtPz * sqrtPz) % self.g_poly
                    _Pz = (sqrtPz * sqrtPz) % self.g_poly
                    assert _Pz == Pz
                    # Find a polynomial sgm(z) = u(z) **2 + z v(z) **2
                    # Find u,v such that v(z) P(z)  =  u(z)  mod g(z)
                    # with deg v <= t//2 and deg u <= (t-1)//2, and deg v minimal
                    uzli, vzli, _ = list_egcd(sqrtPz, self.g_poly)
                    for uzi, vzi in zip(uzli[2:], vzli[2:]):
                        if uzi.degree <= self.t // 2 and vzi.degree <= (self.t -
                                                                        1) // 2:
                            uz = uzi
                            vz = vzi
                            break
                    else:
                        raise RuntimeError(f"Patterson decoding failed.")
                    #uz, vz, _ = galois.egcd(sqrtPz, self.g_poly)
                    sgmz = (uz**2 + galois.Poly.Identity(self.gf) * vz**2
                           )  #% self.g_poly
                    eta = (Sy * sgmz) % self.g_poly
                    d_sgmz = sgmz.derivative() % self.g_poly
                    assert eta == d_sgmz
                    sgmz_roots = sgmz.roots()

                    for ai in sgmz_roots:
                        if int(ai) in self.code_pos:
                            i = self.code_pos[int(ai)]
                            yj[i] ^= 1
                            ej[i] ^= 1
                        else:
                            raise RuntimeError(
                                f"Patterson decoding failed. sgm(z) root ({int(ai)}) not in Goppa code."
                            )
                else:
                    # If P(z) = 0, then Sy(z) = 1/z. This occurs an error on the \alpha=0 bit
                    if 0 not in self.code_pos:
                        raise RuntimeError("Patterson decoding failed.")
                    else:
                        i = self.code_pos[0]
                        yj[i] ^= 1
                        ej[i] ^= 1
                    pass
        return y2, errs

    def decode(self, y: np.ndarray):
        """

        :param y:  [msglen, n] encoded message
        :return:
        """
        y2, e = self.correct_err(y)
        # decode the matrix
        m = f2_matmul(y2, self._decmat.transpose())
        return m, e


def generate_random_goppa(n,
                          t,
                          *,
                          exact_k=False,
                          m=None,
                          rng=None,
                          verbose=False):
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    if m is None:  # pick smallest field that will contain the code
        m = int(np.ceil(np.log2(n)))

    assert m < 64
    #k = n - m * t  # Dimension of the Goppa code
    #assert k > 0
    logging.info(f"Generating random Goppa(n={n}, t={t}) in GF(2^{m})")

    for _ in tqdm(range(100),
                  "Generate Goppa Code",
                  leave=False,
                  disable=not verbose):
        gf = galois.GF(2, m)

        a = gf.primitive_element
        g_poly = _poly_random_search(2**m, t, 'is_irreducible', rng=rng)
        #logging.info(f"Irred poly over GF(2^m): {g_poly}")
        assert n <= 2**m
        if n < 2**m:
            alphs = rng.choice(2**m, size=n, replace=False)
            code_els = gf([a**(w - 1) if w > 0 else 0 for w in alphs])
            code_set = set([int(a) for a in code_els])
            assert len(code_set) == len(code_els)
        else:
            code_els = gf.elements
        #logging.info(f"code elements: {code_els}")

        goppa = GoppaCode(gf, code_els, g_poly)
        if goppa.k == n - t * m or not exact_k:
            logging.info(
                f"Finished building goppa (k={goppa.k}, rate={goppa.k/n}, d={goppa._d})"
            )
            return goppa
        else:
            logging.info(
                f"Rejected Goppa code with  (k={goppa.k}=n-tm+{goppa.k - n + t * m})"
            )

    raise RuntimeError("Goppa code generation failed")
