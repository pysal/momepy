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
        elif item.endswith('.zip'):
            data_files.append(os.path.join("datasets", item))

setup(name='momepy',
      version='0.1a1',
      description='Urban Morphology Measuring Toolkit',
      license="MIT",
      author='Martin Fleischmann',
      author_email='martin@martinfleischmann.net',
      url='https://github.com/martinfleis/momepy',
      packages=['momepy', 'momepy.datasets'],
      package_data={'momepy': data_files},
      )
