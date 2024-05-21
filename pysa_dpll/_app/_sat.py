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
from typer import Typer, Argument, Option, BadParameter, Context
from typing_extensions import Annotated, Optional
from pysa_dpll.sat import optimize
from pysa_dpll.sat.utils import loads
from sys import stdin, stderr
from warnings import warn
from rich import print
import json
from ._globals import __params, print_params

__all__ = ['sat']


def load_cnf(filename: str):
    # Check filename
    if not filename and stdin.isatty():
        raise BadParameter("A CNF must be passed from stdin. "
                           "Otherwise, provide CNF using '-f' / '--file'.")

    # Load cnf
    with (open(filename) if filename else stdin) as file_:
        return loads(file_.read())


# SAT
def sat(max_n_unsat: Annotated[
    int,
    Option("--max-n-unsat",
           "-m",
           help="Maximum number of unsatisfied clauses that are allowed.")] = 0,
       ):
    # Update parameters
    __params.update(locals())

    # Print parameters
    if (__params['verbose']):
        print_params()

    # Load cnf
    cnf_ = load_cnf(__params['filename'])

    # Optimize
    collected_, branches_ = optimize(cnf_,
                                     max_n_unsat=__params['max_n_unsat'],
                                     n_threads=__params['n_threads'],
                                     walltime=__params['walltime'],
                                     verbose=__params['verbose'])

    # At the moment, branches cannot be saved
    if len(branches_):
        warn("At the moment, partial branches cannot be exported", UserWarning)

    # Dump results
    print(
        json.dumps(list(
            map(lambda x: dict(state=str(x.state), n_unsat=x.n_unsat),
                collected_)),
                   indent=2))
