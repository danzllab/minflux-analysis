# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:24:49 2023

@author: jvorlauf_admin
"""

import cython

from math import pi, e

cimport numpy as np
import numpy as np
np.import_array()
DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef np.ndarray _nudft_cython_cdef(np.ndarray t, np.ndarray y, np.ndarray f):
    cdef int i = 0, k = 0
    cdef complex [5000] ft
    cdef complex ft_coeff
    
    # ft = np.zeros(f.shape, dtype=np.float)
    for k in  range(f.size):
        ft_coeff = 0 + 0j
        for i in range(t.size):
            ft_coeff += y[i]*e**(-2*pi*1j*t[i]*f[k])
        
        ft[k] = ft_coeff
        # ft_coeff = y*np.exp(-2*np.pi*1j*t*f[k])
        # ft[k] = np.trapz(ft_coeff, f)
        
    return ft

def _nudft_cython(np.ndarray t, np.ndarray y, np.ndarray f):
    return _nudft_cython_cdef(t, y, f)

