import sys
import setuptools
from pathlib import Path

from cmake_build_extension import CMakeExtension, BuildExtension

init_py = ""

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
                f"-DPYMODULES=ON",
                "-DCALL_FROM_SETUP_PY:BOOL=ON",
            ],
            cmake_component="bindings",
        ),
    ],
    cmdclass=dict(build_ext=BuildExtension)
)