#!/usr/bin/env python

from setuptools import setup

setup(name='ibc_public',
      version='0.1',
      description='Public code for IBC data analysis',
      url='https://github.com/hbp-brain-charting/public_analysis_code',
      author='Bertrand Thirion',
      author_email='bertrand.thirion@inria.fr',
      packages=['ibc_public'],
      #package_data={'ibc_pulic': ['ibc_data/*.tsv']},
      #include_package_data=True,
)
