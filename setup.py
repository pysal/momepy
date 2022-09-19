#!/usr/bin/env/python
import os
import sys

import versioneer
from setuptools import setup

# ensure the current directory is on sys.path so versioneer can be imported
# when pip uses PEP 517/518 build rules.
# https://github.com/python-versioneer/python-versioneer/issues/193
sys.path.append(os.path.dirname(__file__))

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
