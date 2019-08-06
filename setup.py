#!/usr/bin/env/python
import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# get all data dirs in the datasets module
data_files = []

for item in os.listdir("momepy/datasets"):
    if not item.startswith('__'):
        if os.path.isdir(os.path.join("momepy/datasets/", item)):
            data_files.append(os.path.join("datasets", item, '*'))
        elif item.endswith('.gpkg'):
            data_files.append(os.path.join("datasets", item))

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='momepy',
      version='0.1rc1',
      description='Urban Morphology Measuring Toolkit',
      long_description=long_description,
      long_description_content_type="text/markdown",
      license="MIT",
      author='Martin Fleischmann',
      author_email='martin@martinfleischmann.net',
      keywords=["urban morphology",
                "urban morphometrics",
                "tessellation"],
      url='http://momepy.org',
      packages=['momepy', 'momepy.datasets'],
      package_data={'momepy': data_files},
      classifiers=["Programming Language :: Python :: 3",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent",
                   "Intended Audience :: Science/Research",
                   "Topic :: Scientific/Engineering :: GIS"],
      install_requires=["geopandas>=0.5.1",
                        "networkx>=2.3",
                        "libpysal>=4.0.1",
                        "tqdm>=4.25.0"
                        ]
      )
