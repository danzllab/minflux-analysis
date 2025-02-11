# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:27:01 2023

@author: jvorlauf_admin
"""


class MinfluxParameters:
    def __init__(self, sample_type):
        if sample_type == 'blink':
            self._load_parameters_blink()
        elif sample_type == 'bead':
            self._load_parameters_bead()
        else:
            raise ValueError("sample_type must be 'blink' or 'bead'")
        
    
    def _load_parameters_blink(self):
        self.file_name = 'minflux_data/20221223_0201'
        self.n_tcp = 4
        self.t_tcp = 200e-6 + 10e-6     # 10 us added to exposure time for EOD movement 
        self.L = 140
        self.fwhm = 350
        
        self.d_grid = 0
        self.n_grid = 1
       
        self.estimator = 'lms'
        self.filter_type = 'box'
        self.subtract_bg = False
        self.subtract_drifts = -1
        
        self.count_threshold = [35, 70]
        self.t_threshold = 0.05
        self.variation_threshold = 0.15 
        self.cfr_max = 0.7
        self.t_filter = 50
        self.k_bin = 1
        self.n_photon = 4000
        
        
    def _load_parameters_bead(self):
        self.file_name = 'minflux_data/20221006_0401'
        self.n_tcp = 4
        self.t_tcp = 200e-6 + 10e-6     # 10 us added to exposure time for EOD movement 
        self.L = 500
        self.fwhm = 360
        
        self.d_grid = 0
        self.n_grid = 1
       
        self.estimator = 'lms'
        self.filter_type = 'box'
        self.subtract_bg = False
        self.subtract_drifts = -1
        
        self.count_threshold = [400, 1e9]
        self.t_threshold = 0.015
        self.variation_threshold = 1  
        self.cfr_max = 1
        self.t_filter = 1
        self.k_bin = 1
        self.n_photon = 1e6
        
        
    def save_all(self):
        with open(self.file_name + '_params.txt', 'w') as f:
            for param in self.__dict__:
                f.write(f'{param}: {self.__dict__[param]}\n')
                
                
                