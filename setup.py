#!/usr/bin/env python

from distutils.core import setup

setup(name='jointmap',
      version='0.0.1',
    packages=['jointmap'],
    install_requires=['numpy', 'pyfftw'],
    requires=['numpy', 'lenscarf', 'plancklens']
     )