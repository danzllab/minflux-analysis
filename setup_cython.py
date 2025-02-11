# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:49:33 2023

@author: jvorlauf_admin
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize('cython_utils.pyx', annotate=True),
      include_dirs=[numpy.get_include()])
    
    