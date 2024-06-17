import numpy as np

from pysa_stern.bindings import SternOpts, sterncpp_main
from pysa_stern.mld import MLDProblem

def pysa_sternx(mld_problem: MLDProblem, opts: SternOpts):
    pysa_stern_sol = sterncpp_main(mld_problem, opts)
    if pysa_stern_sol is not None:
        return np.asarray(pysa_stern_sol)
    else:
        return None
