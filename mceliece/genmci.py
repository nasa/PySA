#!/bin/env python
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

import fire
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from string import Template

from mceliece.f2mat import f2_matmul
from mceliece.generator import McElieceInstance, NiederInstance
from mceliece.goppa import generate_random_goppa
from mceliece.lincodes import dual_code
from mceliece.randutil import random_weight_t
from mceliece.reduction import to_xorsat, CNFXorSat
from mceliece.tseytin.reduce import reduce_mld_to_sat


def genmci(n: int,
           t: int,
           num_inst: int,
           pub_dir: str,
           priv_dir: str,
           file_pattern: str = "${T}_n${n}_t${t}_m${m}_k${k}",
           f2_deg: int = None,
           nieder: bool = False,
           seed: int = None,
           overwrite: bool = False,
           verbose: bool = False):
    """
    Generates McEliece Cryptographic Instances (MCI).

    Args:
        n: Code length.
        t: Number of correctable errors (Goppa polynomial degree).
        num_instances: Number of instances to generate.
        pub_dir: Public destination directory.
        priv_dir: Private destination directory.
        file_pattern: Pattern to use to create files
                      (default="${T}_n${n}_t${t}_m${m}_k${k}).
        f2_deg: Extension field degree (m). Defaults to the smallest degree that
                will contain a code of size n.
        nieder: Generate McEliece-Niederreiter instances instead of McEliece
                instances.
        seed: Seed for the random number generator. Will seed
              non-deterministically if not provided.
        overwrite: If `True`, files will be overwritten.
        verbose: Verbose output.
    """

    if n <= 1:
        raise RuntimeError(f"Code length n must be at least 2 (Got {n}).")
    if t <= 0:
        raise RuntimeError(
            f"Correctable distance t must be positive (Got {t}).")
    if num_inst <= 0:
        raise RuntimeError(
            f"Required a valid number of instances to generate (Got {num_inst})."
        )
    if f2_deg is not None and f2_deg <= 1:
        raise RuntimeError(f"Extension field degree must be at least 1.")

    m = f2_deg if f2_deg is not None else int(np.ceil(np.log2(n)))
    kcode = n - t * m
    if kcode <= 0:
        raise RuntimeError(
            f"Invalid Goppa code parameters. Size of the codeword (n), number of errors (t) "
            f"and field degree (m) must follow: n - t*m > 0. (Got n={n}, t={t}, m={m})"
        )
    filetmp = Template(file_pattern)
    filepref = filetmp.substitute({
        "T": 'mnci' if nieder else 'mci',
        "n": n,
        "t": t,
        "m": m,
        "k": kcode
    })
    pub_dir = Path(pub_dir)
    priv_dir = Path(priv_dir)

    pub_dir.mkdir(exist_ok=True)
    priv_dir.mkdir(exist_ok=True)

    # Raise an error if file already exists
    if (not overwrite):
        for inst_idx in range(num_inst):
            inst_file = (filepref + f"_{inst_idx}")
            for filename in [
                    pub_dir / (inst_file + "_mld.txt"),
                    pub_dir / (inst_file + "_pcd.txt"),
                    pub_dir / (inst_file + "_xor.cnf"),
                    pub_dir / (inst_file + "_3sat.cnf"),
                    pub_dir / (inst_file + "_xor_3sat.cnf"),
                    pub_dir / (inst_file + "_p3.txt"),
                    priv_dir / (inst_file + "_plain.txt"),
                    priv_dir / (inst_file + "_errs.txt")
            ]:
                if (Path(filename).is_file()):
                    raise FileExistsError(f"File '{filename}' already exists.")

    rng = np.random.Generator(np.random.PCG64(seed))
    for inst_idx in tqdm(range(num_inst),
                         desc='Generating Random Instances',
                         disable=not verbose):
        inst_file = (filepref + f"_{inst_idx}")
        goppa_code = generate_random_goppa(n,
                                           t,
                                           m=m,
                                           rng=rng,
                                           exact_k=True,
                                           verbose=verbose)
        err = np.reshape(random_weight_t(n, t, rng), (1, n))
        if nieder:
            instance = NiederInstance(goppa_code, rng)
            pub, priv = instance.public_private_pair()
            G = pub.Hp
            y = pub.encode(err)
            x = err
        else:
            instance = McElieceInstance(goppa_code, rng)
            pub, priv = instance.public_private_pair()
            mesg = rng.integers(0, 2, (1, goppa_code.k), dtype=np.int8)
            y0 = pub.encode(mesg)
            y = (y0 + err) % 2
            G = pub.Gp.transpose()
            x = mesg
        xsform = to_xorsat(G)
        xsize = len(
            x[0])  # size of the plain text message (k mceliece, n niederreiter)
        nbytes = 1 + xsize // 8 - (1 if xsize % 8 == 0 else 0)
        bytesarr = np.zeros(nbytes, dtype=np.uint8)
        for i in range(xsize):
            j = i % 8
            k = i // 8
            if x[0, i]:
                bytesarr[k] |= (1 << j)
        with open(priv_dir / (inst_file + "_plain.txt"), 'w') as f:
            for i in x[0]:
                f.write(f"{i} ")
            f.write("\n")
            for k in range(nbytes):
                f.write(f"{bytesarr[k]:02X} ")
            f.write('\n')
            if not nieder:
                for i in y0[0]:
                    f.write(f"{i} ")
                f.write("\n")

        with open(priv_dir / (inst_file + "_errs.txt"), 'w') as f:
            for i in range(n):
                if err[0, i] > 0:
                    f.write(f"{i} ")

        with open(pub_dir / (inst_file + "_mld.txt"), 'w') as f:
            if nieder:
                f.write(f'h {n} {len(xsform)} 1 {t}\n')
            else:
                f.write(f'g {kcode} {len(xsform)} 1 {t}\n')
            for l, yi in zip(xsform, y[0]):
                if yi == 0:
                    f.write("-")
                for i in l:
                    f.write(f"{i} ")
                f.write("\n")
        if not nieder:
            Hdual = dual_code(pub.Gp)
            ydual = f2_matmul(y, Hdual.transpose())
            xsform_dual = to_xorsat(Hdual, ydual[0])
            with open(pub_dir / (inst_file + "_pcd.txt"), 'w') as f:
                f.write(f'h {n} {len(xsform_dual)} 1 {t}\n')
                for l in xsform_dual:
                    for i in l:
                        f.write(f"{i} ")
                    f.write("\n")
        with open(pub_dir / (inst_file + "_xor.cnf"), 'w') as f:
            f.write(f"p cnf {n} {len(xsform)}\n")
            for l, yi in zip(xsform, y[0]):
                f.write("x")
                if yi == 0:
                    f.write("-")
                for i in l:
                    f.write(f"{i} ")
                f.write("0\n")
        if nieder:
            # write out mixed XORSat-3SAT problem
            xs_dimacs = reduce_mld_to_sat(G, y[0], t)
            with open(pub_dir / (inst_file + "_xor_3sat.cnf"), 'w') as f:
                f.write(xs_dimacs)
            # write out pure 3SAT problem
            sat_dimacs = reduce_mld_to_sat(G, y[0], t, as_3sat_only=True)
            with open(pub_dir / (inst_file + "_3sat.cnf"), 'w') as f:
                f.write(sat_dimacs)
        if not nieder:
            cnf = CNFXorSat(goppa_code.k, xsform, y[0])
            p3isn = cnf.to_p3_xorsat()
            with open(pub_dir / (inst_file + "_p3.txt"), 'w') as f:
                f.write(f"{p3isn.offset}\n")
                for (i, K) in p3isn.lin:
                    f.write(f"{i} {K}\n")
                for (i, j, K) in p3isn.quad:
                    f.write(f"{i} {j} {K}\n")
                for (i, j, k, K) in p3isn.cub:
                    f.write(f"{i} {j} {k} {K}\n")


def genmci_main():
    fire.Fire(genmci)
