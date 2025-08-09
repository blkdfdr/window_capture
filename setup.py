from setuptools import setup
from setuptools.extension import Extension
import numpy


# Define the extension module
ext_modules = [
    Extension(
        name="window_capture",
        sources=["src/window_capture.cpp"],
        include_dirs=[numpy.get_include()],
        libraries=["Gdiplus", "User32", "Gdi32"],
    )
]

setup(
    name="window_capture",
    version="0.1.0",
    ext_modules=ext_modules,
    install_requires=[
        'numpy',
    ],
    setup_requires=[
        'numpy',
    ],
)