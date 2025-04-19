from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "fwht",
        ["transformation_final.cpp"],
        define_macros=[('VERSION_INFO', "0.0.1")],
    ),
]

setup(
    name="fwht",
    version="0.0.1",
    author="Tomas Antal",
    description="A pybind11 module for inplace fast Walsh-Hadamard transformation",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
