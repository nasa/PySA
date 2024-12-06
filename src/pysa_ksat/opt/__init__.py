import sys

from pysa_ksat import KSatAdvancedException

try:
    import pysa2_cdcl.bindings
    import pysa_dpll.sat
    import pysa_walksat.bindings
except ImportError as e:
    raise KSatAdvancedException from e
