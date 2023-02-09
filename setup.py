#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
"""
    Setup file for jdrones.
    Use setup.cfg to configure your project.
"""
import sys

from distutils.core import setup
from setuptools import find_packages

if __name__ == "__main__":
    packages = find_packages(where="src")
    try:
        setup(
            packages=packages,
            package_dir={"": "src"},
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "and setuptools_scm:\n"
            "   pip install -U setuptools setuptools_scm"
            "\n\n",
            file=sys.stderr,
        )
        raise
