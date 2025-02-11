# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:49:29 2023

@author: jvorlauf_admin
"""

import numpy as  np
import pandas as pd

from minflux_parameters import MinfluxParameters

class DataMinflux:
    
    def __init__(self, parameters, n_cols=7, cols_counts=[0, 4], n_read=-1, offset=1080):
        
        ## offsets for datasets:
        # 20221223_0201: 1486
        # 20221006_0401: 1080
        
        data_bin = np.fromfile(parameters.file_name + ".4P_mfx", offset=offset, count=n_read*n_cols, dtype=np.int64).reshape((-1,n_cols))
        # data_bin = data_bin[data_bin[:,4] < 400]
        self.counts_raw = data_bin[:,cols_counts[0]:cols_counts[1]]
        # counts = counts - 300 #np.where(counts > 30, counts - 30, np.zeros(counts.shape))
        # counts_sim_col = np.hstack((200*np.ones(400), np.zeros(1000), 100*np.ones(300), np.zeros(500)))
        # counts_sim = np.vstack((counts_sim_col, )*4).transpose()
        self.ind_tile = data_bin[:,cols_counts[1]]
        
        self.parameters = parameters
            
        
    def create_grid(self):
        '''
        

        Args:
            d_tiles (TYPE): DESCRIPTION.
            n_tiles (TYPE): DESCRIPTION.

        Returns:
            None.

        '''
        n_tiles = self.parameters.n_grid
        d_tiles = self.parameters.d_grid
        if type(n_tiles) is int:
            n_grid = n_tiles**2
            n_tiles = [n_tiles, n_tiles]
        else:
            n_grid = n_tiles[0]*n_tiles[1]
        if not hasattr(d_tiles, '__iter__'):
            d_tiles = [d_tiles, d_tiles]
        
        n_tiles_data = max(self.ind_tile) + 1
        if n_tiles_data < n_grid:
            raise ValueError(f'Wrong "n_grid" parameter. Data contains {n_tiles_data} tiles.')
        
        self.ind_select = [self.ind_tile % n_grid == i for i in range(n_grid)]
        
        self.counts_tiles = [self.counts_raw[ind_exp] for ind_exp in self.ind_select]
        ymesh, xmesh = np.meshgrid(d_tiles[1]*np.arange(-(n_tiles[1] - 1)/2, (n_tiles[1] + 1)/2),
                                   d_tiles[0]*np.arange(-(n_tiles[0] - 1)/2, (n_tiles[0] + 1)/2))
        self.offsets_tiles = np.vstack([ymesh.flatten(), xmesh.flatten()]).transpose()
        
    
    def create_timestamps(self):
        '''
        

        Args:
            t_tcp (TYPE): DESCRIPTION.

        Returns:
            None.

        '''
        if hasattr(self, 'ind_select'):
            t = self.parameters.t_tcp*np.linspace(0, self.counts_raw.size - self.parameters.n_tcp, self.counts_raw.shape[0])
            t_exp = [t[ind_exp] for ind_exp in self.ind_select]
        else:
            t_exp = [self.parameters.t_tcp*np.linspace(0, self.counts_raw[0].size - self.n_tcp, self.counts_raw[0].shape[0])]
            for c_exp in self.counts_raw[1:]:
                t_exp.append(t_exp[-1][-1] + 
                             self.parameters.t_tcp * np.linspace(0, c_exp.size - self.n_tcp, c_exp.shape[0]))
            
        self.t_exp = t_exp
        
    
    def create_target_coordinates(self, shape='circle', ind_center=3):
        
        if shape == 'circle': 
            if ind_center != None:
                n_circ = self.parameters.n_tcp - 1
                coord_circ = [[self.parameters.L/2*np.cos(2*np.pi*i/n_circ), self.parameters.L/2*np.sin(2*np.pi*i/n_circ)] 
                              for i in range(1, n_circ+1)]
                coord_circ = np.array(coord_circ)
                coord = np.vstack([coord_circ[:ind_center], [0, 0], coord_circ[ind_center:]])
            else:
                n_circ = self.parameters.n_tcp
                coord_circ = [[self.parameters.L/2*np.cos(2*np.pi/n_circ*i/n_circ), self.parameters.L/2*np.sin(2*np.pi*i/n_circ)] 
                              for i in range(1, n_circ+1)]
                coord = np.array(coord_circ)
                
        self.target_coordinates = coord
    
    
    def get_localization_data(self, localization_object):
        self.counts_processed = localization_object.counts_processed
        self.localizations_raw = localization_object.localizations_raw
        self.localizations = localization_object.localizations
        self.t_mask = localization_object.t_mask
        self.count_mask = localization_object.count_mask
        
    
    def save_data(self):
        pass
    
        
    def find_offset(self, offset_range=[0, 3000], n_cols=7):
        for i in range(*offset_range):
            d = np.fromfile(self.parameters.file_name + ".4P_mfx", offset=i, count=n_cols*10, dtype=np.int64).reshape((-1,n_cols))
            dmax = np.max(d[:,:4])   
            if dmax < 10000: print(i, dmax)
        
        
    
class DataSampleLock:
    def __init__(self, file_name):
        pass
    
    
class DataBeamTracking:
    def __init__(self, file_name):
        pass   
    
    
    
    