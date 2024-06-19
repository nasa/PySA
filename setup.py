import os
from pathlib import Path
import setuptools
import sys

from cmake_build_extension import CMakeExtension, BuildExtension
# Check if MPI is needed
PYSA_USE_MPI = 'PYSA_USE_MPI' in os.environ

if PYSA_USE_MPI:
    mpi_conf = ["-DMPI=ON"]
    mpi_req = ["mpi4py"]
    print("Installing PySA-Stern with MPI.")
else:
    mpi_conf = []
    mpi_req = []
    print("Installing PySA-Stern without MPI.")


setuptools.setup(
    ext_modules=[
        CMakeExtension(
            name="PySA-Stern",
            install_prefix="pysa_stern",
            cmake_depends_on=["pybind11"],
            source_dir=str(Path(__file__).parent.absolute()),
            cmake_configure_options=[
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython_ROOT_DIR={Path(sys.prefix)}",
                "-DPYMODULES=ON",
                "-DCALL_FROM_SETUP_PY:BOOL=ON",
            ] + mpi_conf,
            cmake_component="bindings",
        ),
    ],
    cmdclass=dict(build_ext=BuildExtension),
    requires=["numpy"] + mpi_req
)