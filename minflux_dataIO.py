# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:49:29 2023

@author: jvorlauf_admin
"""

import numpy as  np


class DataMinflux:
    '''
    Initialize minflux data object with minflux-parameters and characteristics depending on format of raw data.
    Loads data from ".4P_mfx" file type, but can be easily adapted to other raw data formats.

    Parameters
    ----------
    parameters : minflux parameter profile;
    n_cols : number of saved columns per minflux cycle in raw data file.
    cols_counts : specifies range of columns containing photon counts.
    col_tile : column containing index of tip/tilt mirror position.
    n_read : how many minflux-cycles to read; negative values read entire file.
    offset : how many lines to skip at beginning of file (corresponding to metadata, depends on file format).

    '''
    def __init__(self, parameters, n_cols=7, cols_counts=[0, 4], col_tile=4, n_read=-1, offset=1080):
        ## offsets for datasets:
        # 20221223_0201: 1486
        # 20221006_0401: 1080
        
        data_bin = np.fromfile(parameters.file_name + ".4P_mfx", offset=offset, count=n_read*n_cols, dtype=np.int64).reshape((-1,n_cols))
        self.counts_raw = data_bin[:,cols_counts[0]:cols_counts[1]]
        self.ind_tile = data_bin[:,col_tile]
        
        self.parameters = parameters
            
        
    def create_grid(self):
        '''
        Create grid of tip/tilt mirror positions as specified in parameters.
        '''
        n_tiles = self.parameters.n_grid
        d_tiles = self.parameters.d_grid
        if type(n_tiles) is int:    # if n_tiles is integer, create equal number of points for x and y
            n_grid = n_tiles**2     # n_grid: total number of points
            n_tiles = [n_tiles, n_tiles]    # n_tiles: number of points for each axis
        else:   
            n_grid = n_tiles[0]*n_tiles[1]
        if not hasattr(d_tiles, '__iter__'):    # if d_tiles given as single value, set equal for x and y
            d_tiles = [d_tiles, d_tiles]
        
        n_tiles_data = max(self.ind_tile) + 1
        if n_tiles_data < n_grid:   # assert that number of positions specified in parameters does not exceed positions in data 
            raise ValueError(f'Wrong "n_grid" parameter. Data contain {n_tiles_data} tiles.')
        
        self.ind_select = [self.ind_tile % n_grid == i for i in range(n_grid)]  # group localizations to corresponding tip/tilt positions (if grid is rescanned, ind_tile increases beyond n_grid)
        
        self.counts_tiles = [self.counts_raw[ind_exp] for ind_exp in self.ind_select]   # assign counts to corresponding tiles (tip/tilt positions)
        ymesh, xmesh = np.meshgrid(d_tiles[1]*np.arange(-(n_tiles[1] - 1)/2, (n_tiles[1] + 1)/2),
                                   d_tiles[0]*np.arange(-(n_tiles[0] - 1)/2, (n_tiles[0] + 1)/2))
        self.offsets_tiles = np.vstack([ymesh.flatten(), xmesh.flatten()]).transpose()      # generate coordinate offsets to be applied to localizations
        
    
    def create_timestamps(self):
        '''
        Create time stamps for count traces.
        '''
        t = self.parameters.t_tcp*np.linspace(0, self.counts_raw.size - self.parameters.n_tcp, self.counts_raw.shape[0])
        if hasattr(self, 'ind_select'):     # if grid was created, create list of time stamps starting from begin of measurement for every position/experiment
            t_exp = [t[ind_exp] for ind_exp in self.ind_select]
        else:   # if no grid was created, create time stamps directly
            t_exp = [t]
            
        self.t_exp = t_exp
        
    
    def create_target_coordinates(self, shape='circle', ind_center=3):
        '''
        Create TCP corrdinates. Currently, exposures along circle are implemented. If given, an additional exposure is added in the center of the circle.

        Parameters
        ----------
        shape : shape of TCP; currently, only "circle" implemented.
        ind_center : index of column corresponding to central exposure; if None, it is assumed that all TCP exposures lie on cirle.
        '''
        if shape == 'circle': 
            if ind_center != None:
                n_circ = self.parameters.n_tcp - 1  # all except 1 exposure lie on circle
                coord_circ = [[self.parameters.L/2*np.cos(2*np.pi*i/n_circ), self.parameters.L/2*np.sin(2*np.pi*i/n_circ)] 
                              for i in range(1, n_circ+1)]
                coord_circ = np.array(coord_circ)
                coord = np.vstack([coord_circ[:ind_center], [0, 0], coord_circ[ind_center:]])   # insert central exposure with coordinates (0, 0)
            else:
                n_circ = self.parameters.n_tcp  # all exposures lie on circle
                coord_circ = [[self.parameters.L/2*np.cos(2*np.pi/n_circ*i/n_circ), self.parameters.L/2*np.sin(2*np.pi*i/n_circ)] 
                              for i in range(1, n_circ+1)]
                coord = np.array(coord_circ)
        
        else:
            raise ValueError('Currently only target coordinates of shape = "circle" are implemented.')
                
        self.target_coordinates = coord
    
    
    def get_localization_data(self, localization_object):
        '''Load localization data to DataMinflux object for visualization and analysis.'''
        self.counts_processed = localization_object.counts_processed
        self.localizations_raw = localization_object.localizations_raw
        self.localizations = localization_object.localizations
        self.t_mask = localization_object.t_mask
        self.count_mask = localization_object.count_mask
        
    
    def save_data(self):
        '''To be implemented.'''
        pass
    
        
    def find_offset(self, offset_range=[0, 3000], n_cols=7, cols_counts=[0, 4], counts_max=1024):
        '''Utility function for finding reasonable offsets corresponding to metadata in raw data file. First hit likely to be correct offset.
        Based on assumption that counts should be reasonably small values (wrong offset will result in arbitrary values within range of np.int64).
       
        Parameters
        ----------
        offset_range : range of offsets to test.
        n_cols : number of saved columns per minflux cycle in raw data file.
        cols_counts : specifies range of columns containing photon counts.
        counts_max : maximum count number for offset to be considered a hit.
        '''
        for i in range(*offset_range):
            d = np.fromfile(self.parameters.file_name + ".4P_mfx", offset=i, count=n_cols*10, dtype=np.int64).reshape((-1,n_cols))  # for every offset in specified range, read first 10 iterations from file
            dmax = np.max(d[:,cols_counts[0]:cols_counts[1]])       # calculate maximum of columns that should contain photon counts.
            dmin = np.min(d[:,cols_counts[0]:cols_counts[1]])       # calculate minimum of columns that should contain photon counts.
            if dmin >= 0 and dmax < counts_max:   # if given offset yields non-negative values with reasonably small max., print to indicate potential hit
                print(f'For offset = {i}, photon counts go up to {dmax}.')    
        
        
# load active sample stabilization data; to be implemented
class DataSampleLock:
    def __init__(self, file_name):
        pass
    
# load beam tracking data; to be implemented
class DataBeamTracking:
    def __init__(self, file_name):
        pass   
    
    
    
    