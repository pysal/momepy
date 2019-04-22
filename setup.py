#!/usr/bin/env/python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='momepy',
      version='0.1a1',
      description='Urban Morphology Measuring Toolkit',
      license="MIT",
      author='Martin Fleischmann',
      author_email='martin@martinfleischmann.net',
      url='https://github.com/martinfleis/momepy',
      packages=['momepy'],
      )
