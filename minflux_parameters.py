# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:27:01 2023

@author: jvorlauf_admin
"""


class MinfluxParameters:
    def __init__(self, sample_type):
        ''''Initialization with a given sample_type loads a bespoke parameter profile. Currently "blink" and "bead" implemented.'''
        if sample_type == 'blink':
            self._load_parameters_blink()
        elif sample_type == 'bead':
            self._load_parameters_bead()
        else:
            raise ValueError("sample_type must be 'blink' or 'bead'")
        
    
    def _parameter_template(self):
        self.file_name = 'minflux_data/'    # path to raw data file
        self.n_tcp = 4  # number of TCP-positions per cycle
        self.t_tcp = 0.0 + 10e-6     # exposure time per TCP-position in s; 10 us added to exposure time for EOD movement (set in hardware control software)
        self.L = 1.0    # TCP-diameter/-size in nm
        self.fwhm = 1.0     # ful-width-at-half-maximum of the minflux-beam in nm
        
        self.d_grid = 0.0   # spacing of adjacent positions of the tip/tilt mirror in nm; float or duple of floats for different axes
        self.n_grid = 1   # number of tip/tilt mirror positions ("experiments"); int or duple of ints for different axes
       
        self.estimator = ''     # position estimator; "mle", "lms", "mlms" are implemented
        self.filter_type = ''   # type of filter to remove noise; "box", "gauss", "triangle" are implemented
        self.subtract_bg = False    # boolean indicating if background should be subtracted from count traces before localization; not recommended - background should be modeled, not subtracted!
        self.subtract_drifts = -1   # degree of polynomial to be fit for post hoc drift correction
        
        self.count_threshold = [1.0, 2.0]   # lower and upper count threshold for extracting single-molecule emission events
        self.t_threshold = 0.0      # minimum duration of emission events, in s
        self.variation_threshold = 0.0  # to exclude time points with high variance (e.g., during on- and off-switching, fluctuations) relative to count levels; 0.1-0.2 found to work well
        self.cfr_max = 0.0  # maximum center frequency ratio; more useful for iterative minflux; so far used mostly >= 0.5 
        self.t_filter = 1   # time constant of filter; width of box, sigma of gauss, or rise-/fall-time for triangle
        self.k_bin = 1      # integer factor for temporal binning of count traces; mostly remained 1 and used filter to increase SNR
        self.n_photon = -1  # extracted emission events get binned until n_photon overall counts are reached; negative numbers: no binning
        
        
    def _load_parameters_blink(self):
        self.file_name = 'minflux_data/20221223_0201'
        self.n_tcp = 4
        self.t_tcp = 200e-6 + 10e-6     # 10 us added to exposure time for EOD movement 
        self.L = 140
        self.fwhm = 350
        
        self.d_grid = 0
        self.n_grid = 1
       
        self.estimator = 'mle'
        self.filter_type = 'box'
        self.subtract_bg = False
        self.subtract_drifts = 11
        
        self.count_threshold = [25, 65]
        self.t_threshold = 0.05
        self.variation_threshold = 0.15 
        self.cfr_max = 0.5
        self.t_filter = 45
        self.k_bin = 1
        self.n_photon = 4000
        
        
    def _load_parameters_bead(self):
        self.file_name = 'minflux_data\\20221216_0102_auBeads_minflux_tiling_20x20grid_2rescans'
        self.n_tcp = 4
        self.t_tcp = 200e-6 + 10e-6     # 10 us added to exposure time for EOD movement 
        self.L = 500
        self.fwhm = 360
        
        self.d_grid = 20
        self.n_grid = 20
       
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
                
                
                