#!/usr/bin/env/python
import os
import sys

import versioneer
from setuptools import setup

# ensure the current directory is on sys.path so versioneer can be imported
# when pip uses PEP 517/518 build rules.
# https://github.com/python-versioneer/python-versioneer/issues/193
sys.path.append(os.path.dirname(__file__))


# get all data dirs in the datasets module
data_files = []

for item in os.listdir("momepy/datasets"):
    if not item.startswith("__"):
        if os.path.isdir(os.path.join("momepy/datasets/", item)):
            data_files.append(os.path.join("datasets", item, "*"))
        elif item.endswith(".gpkg"):
            data_files.append(os.path.join("datasets", item))

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name="momepy",
    version=versioneer.get_version(),
    description="Urban Morphology Measuring Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD",
    author="Martin Fleischmann",
    author_email="martin@martinfleischmann.net",
    keywords=["urban morphology", "urban morphometrics", "tessellation"],
    url="http://momepy.org",
    packages=["momepy", "momepy.datasets"],
    package_data={"momepy": data_files},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    install_requires=[
        "geopandas>=0.8.0",
        "networkx>=2.3",
        "libpysal>=4.2.0",
        "tqdm>=4.27.0",
        "pygeos",
        "packaging",
    ],
    cmdclass=versioneer.get_cmdclass(),
)
