#!/usr/bin/env python
import os
import sys

from setuptools import setup  # No need to import Extension here
from Cython.Build import cythonize
import numpy  # Import numpy here


# the current working directory
_cwd = os.path.abspath(os.path.expanduser(os.path.dirname(__file__)))


def ext_modules():
    """Generate a list of extension modules for embreex."""
    if os.name == "nt":
        # embree search locations on windows
        includes = [
            numpy.get_include(),  # Use numpy.get_include()
            "c:/Program Files/Intel/embree4/include",
            # No need for the local embree4/include if the above is correct.
            # os.path.join(_cwd, "embree4", "include"),
        ]
        libraries = [
            "embree4"  # Just the library name, no path here
        ]
        library_dirs = [  # library *directories* go here
            "C:/Program Files/Intel/embree4/lib",
            # os.path.join(_cwd, "embree4", "lib"), # Only if you *really* have a local copy
        ]

    else:
        # embree search locations on posix
        includes = [
            numpy.get_include(), # Use numpy.get_include()
            "/opt/local/include",
             os.path.join(_cwd, "embree4", "include"),

        ]
        libraries = ["embree4"] # Just the library name
        library_dirs = ["/opt/local/lib", os.path.join(_cwd, "embree4", "lib")] # library *directories*

    ext_modules = cythonize(
        "embreex/*.pyx",
        include_path=includes,
        language_level=3,  # Use Python 3 language level
    )
    for ext in ext_modules:
        #ext.include_dirs = includes  # Already set in include_path
        ext.library_dirs = library_dirs  # Correct.
        ext.libraries = libraries      # Correct.
        ext.language = "c++" #<- We add this line
    return ext_modules


def load_pyproject() -> dict:
    """A hack for Python 3.6 to load data from `pyproject.toml`

    The rest of setup is specified in `pyproject.toml` but moving dependencies
    to `pyproject.toml` requires setuptools>61 which is only available on Python>3.7
    When you drop Python 3.6 you can delete this function.
    """
    # this hack is only needed on Python 3.6 and older
    if sys.version_info >= (3, 7):
        return {}

    import tomli

    with open(os.path.join(_cwd, "pyproject.toml"), "r") as f:
        pyproject = tomli.load(f)

    return {
        "name": pyproject["project"]["name"],
        "version": pyproject["project"]["version"],
        "install_requires": pyproject["project"]["dependencies"],
    }


try:
    with open(os.path.join(_cwd, "README.md"), "r") as _f:
        long_description = _f.read()
except BaseException:
    long_description = ""

setup(
    ext_modules=ext_modules(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    **load_pyproject(),
)