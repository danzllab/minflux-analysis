# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:52:28 2023

@author: jvorlauf_admin
"""

import numpy as np
import warnings
from scipy.interpolate import griddata
from scipy.optimize import curve_fit, least_squares
from scipy.signal import fftconvolve



class PSFmodel:
    
    def __init__(self, psf_type, parameters):
        '''
        Initialize PSF-model based on theoretical beam shape.

        Parameters
        ----------
        psf_type : currently only "doughnut" implemented
        parameters : minflux-parameter file
        '''
        self.psf_type = psf_type
        self.parameters = parameters
        
    def create_psf_model(self, grid_size=500, px_size=0.5):
        '''
        Create psf class attribute as as grid of pixel intensity values.

        Parameters
        ----------
        grid_size : size of coordinate grid per axis in nm; currently only square grids supported
        px_size : pixel size of coordinate grid
        '''
        self.grid_size = grid_size
        self.px_size = px_size
        if self.psf_type == 'doughnut':
            self.psf = self._f_doughnut()
        else:
            raise AttributeError('PSF type not supported.')
    
    
    def _f_doughnut(self):
        '''
        Function producing a doughnut PSF (i.e., first-order optical vortex) as grid of intensities.

        Parameters
        ----------
        fwhm : full width as half maximum in nm
        grid_size :  size of coordinate grid per axis in nm; currently only square grids supported
        px_size :  pixel size of coordinate grid

        Returns
        -------
        2D array of pixel intensity values
        '''
        coord = np.arange(-self.grid_size/2, self.grid_size/2, self.px_size)
        xx, yy = np.meshgrid(coord, coord)
        rr2 = xx**2 + yy**2
        f = rr2/self.parameters.fwhm**2 * np.exp(-4*np.log(2)*rr2/self.parameters.fwhm**2)  # values not normalized
        return f




class PSFcalibration:
    
    def __init__(self, frames_raw, px_size, psf_model={'name': 'poly', 'k': 4}):
        self.frames_raw = frames_raw
        self.frames_dim = np.array(frames_raw.shape)
        self.px_size = px_size
        
        # if single frame given, convert to stack of length 1
        if len(self.frames_dim) == 2:
            self.frames_raw = np.array([self.frames_raw])
            self.frames_dim = np.array(self.frames_raw.shape)
        
        self.psf_model = psf_model
        
    def _f_gauss(self, xy, x0=None, y0=None, fwhm=300, intensity=100):
        if x0 == None: 
            x0 = np.mean(xy[0])
        if y0 == None: 
            y0 = np.mean(xy[1])
            
        xx, yy = xy[0], xy[1]
        rr2 = (xx - x0)**2 + (yy - y0)**2
        f = intensity * np.exp(-4*np.log(2)*rr2/fwhm**2)
        return f
    
    def _f_doughnut(self, xy, x0=None, y0=None, fwhm=300, intensity=100):
        if x0 == None: 
            x0 = np.mean(xy[0])
        if y0 == None: 
            y0 = np.mean(xy[1])
            
        xx, yy = xy[0], xy[1]
        rr2 = (xy[0] - x0)**2 + (xy[1] - y0)**2
        f = intensity*4*np.log(2)*np.e*rr2/fwhm**2 * np.exp(-4*np.log(2)*rr2/fwhm**2)
        return f
    
    def _f_poly(self, xy, x0=None, y0=None, p=None, k=4):
        if x0 == None: 
            x0 = np.mean(xy[0])
        if y0 == None: 
            y0 = np.mean(xy[1])
        if type(p) == None:
            # p = np.ones(k*(k+1)/2)
            p = np.ones((k + 1, k + 1))     #not all ps will actually be needed
        elif type(p) != np.ndarray:
            p = np.asarray(p)
            
        p = p.reshape(k + 1, k + 1)     # if used for fitting, p must be handed over as flattened array
        
        xx, yy = xy[0], xy[1]
        f = np.zeros(xx.shape)
        for i in range(k + 1):
            # for j in range(i + 1):
            for j in range(k + 1):
                # f += p[i + k*(i-j)] * (xy[0] - center[0])**i * (xy[1] - center[1])**(i-j)
                # f += p[i, j] * (xx - x0)**i * (yy - y0)**(i-j)
                f += p[i, j] * (xx - x0)**i * (yy - y0)**j
            
        return f
    
    
    def fit_single_frames(self, modelfun=None):
        if modelfun == None:
            modelfun = self.pfs_model
        
        coord = np.meshgrid(np.arange(0, self.px_size*self.frames_dim[1], self.px_size),
                            np.arange(0, self.px_size*self.frames_dim[2], self.px_size))
        coord = np.vstack([coord[0].flatten(), coord[1].flatten()])
        coord_center = np.mean(coord, axis=1)
        
        if modelfun['name'] == 'poly':
            f_fit = lambda xy, x0, y0, *p: self._f_poly(xy, x0, y0, p, k=modelfun['k'])
            frame_avg_init = np.mean(self.frames_raw, axis=0)
            p_init = np.zeros((modelfun['k'] + 1, modelfun['k'] + 1))
            px_center = np.rint(coord_center/self.px_size).astype(int)
            p_init[1,0] = (np.mean(frame_avg_init[[px_center[0]-1, px_center[0]+1], px_center[1]]) - frame_avg_init[px_center[0], px_center[1]])/self.px_size
            p_init[0,1] = (np.mean(frame_avg_init[px_center[0], [px_center[1]-1, px_center[1]+1]]) - frame_avg_init[px_center[0], px_center[1]])/self.px_size
            p0 = [coord_center[0], coord_center[1], *p_init.flatten()]
        elif modelfun['name'] == 'doughnut':
            f_fit = self._f_doughnut
            p0 = [coord_center[0], coord_center[1], 300, 100]
        elif modelfun['name'] == 'gauss':
            f_fit = self._f_gauss
            p0 = [coord_center[0], coord_center[1], 300, 100]
        else:
            raise ValueError("modelfun['name'] must be 'poly', 'doughnut' or 'gauss'.") 
          
        fit_centers = np.zeros((2, len(self.frames_raw)))
        for i in range(len(self.frames_raw)):
            # f_res = lambda p: (f_fit(coord, *p) - self.frames_raw[i])**2
            # fit_res = least_squares(f_res, [15, 15, 10, 100])
            fit_res, cov = curve_fit(f_fit, coord, self.frames_raw[i].flatten(), p0=p0)
            fit_centers[0, i], fit_centers[1, i] = fit_res[0], fit_res[1]
            
            # fig, ax = plt.subplots(1, 2, num=self.plot_ind, clear=True)
            # self.plot_ind += 1
            # ax[0].imshow(self.frames_raw[i])
            # ax[1].imshow(f_fit(coord, *fit_res).reshape(self.frames_dim[1:]))
            
        self.drifts_raw = fit_centers - fit_centers[:,0][:,np.newaxis]
        
        
    def correct_drifts(self, k_smooth = 10, roi=None):
        drifts = self.drifts_raw
        if roi == None:
            drift_pk = np.max(np.abs(drifts), axis=1)
            roi = self.px_size*self.frames_dim[1:] - drift_pk - 1
            
        drifts_smoothed = fftconvolve(drifts, np.ones((1, k_smooth)), axes=1, mode='same')
        drifts_smoothed = np.rint(drifts_smoothed/self.px_size).astype(int)     #divide by px size to switch from "nm" coord sys. to "grid" coord sys.
        roi = np.rint(roi/self.px_size).astype(int)     #divide by px size to switch from "nm" coord sys. to "grid" coord sys.
        
        frames_corrected = []
        for frame, drift_frame in zip(self.frames_raw, drifts_smoothed.transpose()):
            try:
                crop_x = ((self.frames_dim[1] - roi[0])//2 + drift_frame[0], (self.frames_dim[1] + roi[0])//2 + drift_frame[0])
                crop_y = ((self.frames_dim[2] - roi[1])//2 + drift_frame[1], (self.frames_dim[2] + roi[1])//2 + drift_frame[1])
            except IndexError:
                crop_x = ((self.frames_dim[1] - roi)//2 + drift_frame[0], (self.frames_dim[1] + roi)//2 + drift_frame[0])
                crop_y = ((self.frames_dim[2] - roi)//2 + drift_frame[1], (self.frames_dim[2] + roi)//2 + drift_frame[1])
                
            frames_corrected.append(frame[crop_x[0]:crop_x[1],crop_y[0]:crop_y[1]])
            
        frames_corrected = np.asarray(frames_corrected)
        self.frames_processed = frames_corrected
        return frames_corrected


    def psf_fit(self, modelfun=None, frame_avg=None):
        if modelfun == None:
            modelfun = self.psf_model
        if frame_avg == None:
            try:
                frame_avg = self.frame_avg
            except AttributeError:
                frame_avg = np.mean(self.frames_processed, axis=0)
                self.frame_avg = frame_avg
        
        coord = np.meshgrid(np.arange(0, self.px_size*frame_avg.shape[0], self.px_size),
                            np.arange(0, self.px_size*frame_avg.shape[1], self.px_size))
        coord = np.vstack([coord[0].flatten(), coord[1].flatten()])
        coord_center = np.mean(coord, axis=1)
        
        if modelfun['name'] == 'poly':
            f_fit = lambda xy, x0, y0, *p: self._f_poly(xy, x0, y0, p, k=modelfun['k'])
            p_init = np.zeros((modelfun['k'] + 1, modelfun['k'] + 1))
            px_center = np.rint(coord_center/self.px_size).astype(int)
            p_init[1,0] = (np.mean(frame_avg[[px_center[0]-1, px_center[0]+1], px_center[1]]) - frame_avg[px_center[0], px_center[1]])/self.px_size
            p_init[0,1] = (np.mean(frame_avg[px_center[0], [px_center[1]-1, px_center[1]+1]]) - frame_avg[px_center[0], px_center[1]])/self.px_size
            p0 = [coord_center[0], coord_center[1], *p_init.flatten()]
        elif modelfun['name'] == 'doughnut':
            f_fit = self._f_doughnut
            p0 = [coord_center[0], coord_center[1], 350, np.mean(frame_avg)*frame_avg.size]
        elif modelfun['name'] == 'gauss':
            f_fit = self._f_gauss
        else:
            raise ValueError("modelfun['name'] must be 'poly', 'doughnut' or 'gauss'.") 
          
        fit_res, cov = curve_fit(f_fit, coord, frame_avg.flatten(), p0=p0)
        
        return fit_res, f_fit(coord, *fit_res).reshape(frame_avg.shape)
 
        


# copies from analyzeMinfluxData class in minflux_localization_V2
def set_psf(self, img_psf, px_size_img=None):
    if px_size_img == None:
        px_size_img = self.px_size_grid
    img_psf = np.array(img_psf)
    n_px_img = np.array(img_psf.shape)
    img_size = n_px_img*px_size_img
    d_size = img_size - self.grid_size     #amount by which psf image is larger (or smaller) than grid
    if d_size[0] < px_size_img or d_size[1] < px_size_img:
        warnings.warn('PSF image is smaller than grid! This might lead to artifacts for MLE localization!')

    px_x, px_y = np.meshgrid(np.arange(0, img_size[0], px_size_img) + px_size_img/2, 
                             np.arange(0, img_size[1], px_size_img) + px_size_img/2)
    grid_x, grid_y = np.meshgrid(np.arange(0, self.grid_size, self.px_size_grid) + d_size[0]/2,
                                 np.arange(0, self.grid_size, self.px_size_grid) + d_size[1]/2)
    
    # interpolate psf image at grid points
    psf = griddata((px_x.flatten(), px_y.flatten()), img_psf.flatten(), (grid_x, grid_y), method='cubic')
    psf = np.nan_to_num(psf)    #to handle cases where image is smaller than grid
    psf = np.where(psf > 1e-12, psf, np.zeros(psf.shape) + 1e-12)   # don't want <=0 when forming log for MLE
    self.psf = psf


