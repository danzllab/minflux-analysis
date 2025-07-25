# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:52:28 2023

@author: jvorlauf_admin
"""

import numpy as np
import warnings
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve




class PSFmodel:
    '''
    Initialize PSF-model based on theoretical beam shape.

    Parameters
    ----------
    psf_type : currently only "doughnut" implemented
    parameters : minflux-parameter file
    '''    
    def __init__(self, psf_type, parameters):
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
    '''
    PSF-calibration by fitting experimental data to model function.

    Parameters
    ----------
    frames_raw : raw frames as numpy array; either single 2D-image or stack of multiple images to increase contrast
    px_size : pixel size of raw data
    psf_model : type of function to use for modeling PSF; "gauss", "doughnut", "poly" implemented; dictionary with "name" and function-specific parameters (currently only "k" for polynomial order)
    '''
    def __init__(self, frames_raw, px_size, psf_model={'name': 'polynomial', 'k': 4}):
        self.frames_raw = frames_raw
        self.frame_avg = np.mean(frames_raw, axis=0)
        self.frames_dim = np.array(frames_raw.shape)
        self.px_size_img = px_size
        
        # if single frame given, convert to stack of length 1
        if len(self.frames_dim) == 2:
            self.frames_raw = np.array([self.frames_raw])
            self.frames_dim = np.array(self.frames_raw.shape)
        
        self.psf_model = psf_model
        
        
    def _f_gauss(self, xy, x0=None, y0=None, fwhm=300, intensity=100):
        '''
        2D Gaussian function

        Parameters
        ----------
        xy : coordinate mesh
        x0 : peak position along x; specify or assign as center of coordinate mesh
        y0 : peak position along y; specify or assign as center of coordinate mesh
        fwhm : full width at half maximum
        intensity : overall intensity; not normalized, so arbitrary value

        Returns
        -------
        2D array with pixel intensity values
        '''
        if x0 is None: 
            x0 = np.mean(xy[0])
        if y0 is None: 
            y0 = np.mean(xy[1])
            
        xx, yy = xy[0], xy[1]
        rr2 = (xx - x0)**2 + (yy - y0)**2
        f = intensity * np.exp(-4*np.log(2)*rr2/fwhm**2)
        return f
    
    
    def _f_doughnut(self, xy, x0=None, y0=None, fwhm=300, intensity=100):
        '''
        2D "doughnut" function, i.e., first-order optical vortex

        Parameters
        ----------
        xy : coordinate mesh
        x0 : peak position along x; specify or assign as center of coordinate mesh
        y0 : peak position along y; specify or assign as center of coordinate mesh
        fwhm : full width at half maximum
        intensity : overall intensity; not normalized, so arbitrary value

        Returns
        -------
        2D array with pixel intensity values
        '''
        if x0 is None: 
            x0 = np.mean(xy[0])
        if y0 is None: 
            y0 = np.mean(xy[1])
            
        xx, yy = xy[0], xy[1]
        rr2 = (xy[0] - x0)**2 + (xy[1] - y0)**2
        f = intensity*4*np.log(2)*np.e*rr2/fwhm**2 * np.exp(-4*np.log(2)*rr2/fwhm**2)
        return f
    
    
    def _f_polynomial(self, xy, x0=None, y0=None, p=None, k=4):
        '''
        2D polynomial function

        Parameters
        ----------
        xy : coordinate mesh
        x0 : peak position along x; specify or assign as center of coordinate mesh
        y0 : peak position along y; specify or assign as center of coordinate mesh
        p : polynomial coefficients; if none, set all to 1
        k : order of polynomial

        Returns
        -------
        2D array with pixel intensity values
        '''
        if x0 is None: 
            x0 = np.mean(xy[0])
        if y0 is None: 
            y0 = np.mean(xy[1])
        if type(p) is None:
            # p = np.ones(k*(k+1)/2)
            p = np.ones((k + 1, k + 1))     #not all ps will actually be needed
        elif type(p) != np.ndarray:
            p = np.asarray(p)
            
        p = p.reshape(k + 1, k + 1)     # p is handed over to function as flattened array for compatibility with fitting
        
        xx, yy = xy[0], xy[1]
        f = np.zeros(xx.shape)  # initialize array with 0s, then fill up with orders (including cross-terms) one by one
        for i in range(k + 1):
            for j in range(k + 1):
                f += p[i, j] * (xx - x0)**i * (yy - y0)**j
            
        return f
    
    
    def fit_single_frames(self, modelfun=None):
        '''Fit model function to single frames to obtain center positions indicating drifts between frames; generates drifts_raw attribute (2D array).'''
        if modelfun is None:
            modelfun = self.pfs_model
        
        coord = np.meshgrid(np.arange(0, self.px_size_img*self.frames_dim[1], self.px_size_img),
                            np.arange(0, self.px_size_img*self.frames_dim[2], self.px_size_img))
        coord = np.vstack([coord[0].flatten(), coord[1].flatten()])
        coord_center = np.mean(coord, axis=1)
        
        if modelfun['name'] == 'polynomial':
            f_fit = lambda xy, x0, y0, *p: self._f_polynomial(xy, x0, y0, p, k=modelfun['k'])   # format function for fitting
            frame_avg_init = np.mean(self.frames_raw, axis=0)   # average across frames, pixel by pixel
            p_init = np.zeros((modelfun['k'] + 1, modelfun['k'] + 1))
            px_center = np.rint(coord_center/self.px_size_img).astype(int)
            p_init[1,0] = (np.mean(frame_avg_init[[px_center[0]-1, px_center[0]+1], px_center[1]]) - frame_avg_init[px_center[0], px_center[1]])/self.px_size_img       # not sure if overkill to estimate crude gradients at center pixels?
            p_init[0,1] = (np.mean(frame_avg_init[px_center[0], [px_center[1]-1, px_center[1]+1]]) - frame_avg_init[px_center[0], px_center[1]])/self.px_size_img
            p0 = [coord_center[0], coord_center[1], *p_init.flatten()]      # rough guesses for initial parameters
        elif modelfun['name'] == 'doughnut':
            f_fit = self._f_doughnut
            p0 = [coord_center[0], coord_center[1], 300, 100]   # rough guesses for initial parameters
        elif modelfun['name'] == 'gauss':
            f_fit = self._f_gauss
            p0 = [coord_center[0], coord_center[1], 300, 100]   # rough guesses for initial parameters
        else:
            raise ValueError("modelfun['name'] must be 'polynomial', 'doughnut' or 'gauss'.") 
          
        fit_centers = np.zeros((2, len(self.frames_raw)))
        for i in range(len(self.frames_raw)):   # fit every frames in stack to specified function
            fit_res, cov = curve_fit(f_fit, coord, self.frames_raw[i].flatten(), p0=p0)
            fit_centers[0, i], fit_centers[1, i] = fit_res[0], fit_res[1]
                        
        self.drifts_raw = fit_centers - fit_centers[:,0][:,np.newaxis]
        
        
    def correct_drifts(self, k_smooth = 10, roi=None):
        '''
        Use previously obtained drifts_raw to compensate drifts between frames

        Parameters
        ----------
        k_smooth : smoothing factor for drift correction (width of box filter, in frame numbers)
        roi : specify size of region of interest in nm for cropping drift-corrected frames around center; single value for square or list with sizes for x and y; if none, use maximum width compatible with magnitude of drifts (without extrapolating)

        Returns
        -------
        Drift-corrected, cropped frames.
        '''
        drifts = self.drifts_raw
        if roi is None:
            drift_pk = np.max(np.abs(drifts), axis=1)
            roi = self.px_size_img*self.frames_dim[1:] - drift_pk - 1   #maximum ROI fully encompassing all drift-corrected frames
            
        drifts_smoothed = fftconvolve(drifts, np.ones((1, k_smooth)), axes=1, mode='same')      # convolve drift curves with box filter of width k_smooth to remove noise of drift estimation; performed separately for x and y
        drifts_smoothed = np.rint(drifts_smoothed/self.px_size_img).astype(int)     #rount to nearest pixel; divide by px size to switch from "nm" coord system to "pixel" coord system
        roi = np.rint(np.array(roi)/self.px_size_img).astype(int)     # divide by px size to switch from "nm" coord system to "pixel" coord system
        
        frames_corrected = []
        for frame, drift_frame in zip(self.frames_raw, drifts_smoothed.transpose()):
            if hasattr(roi, '__iter__'):    # add drifts and crop to rectangular ROI
                crop_x = ((self.frames_dim[1] - roi[0])//2 + drift_frame[0], (self.frames_dim[1] + roi[0])//2 + drift_frame[0])
                crop_y = ((self.frames_dim[2] - roi[1])//2 + drift_frame[1], (self.frames_dim[2] + roi[1])//2 + drift_frame[1])
            else:   # add drifts and crop to square ROI
                crop_x = ((self.frames_dim[1] - roi)//2 + drift_frame[0], (self.frames_dim[1] + roi)//2 + drift_frame[0])
                crop_y = ((self.frames_dim[2] - roi)//2 + drift_frame[1], (self.frames_dim[2] + roi)//2 + drift_frame[1])
                
            frames_corrected.append(frame[crop_x[0]:crop_x[1],crop_y[0]:crop_y[1]])     # append drift-corrected, cropped frame
            
        frames_corrected = np.asarray(frames_corrected)
        self.frames_processed = frames_corrected
        self.frame_avg = np.mean(frames_corrected, axis=0)
        
        return frames_corrected


    def psf_fit(self, modelfun=None, frame_avg=None):
        '''
        Fit PSF data to model function.

        Parameters
        ----------
        modelfun : model function; dictionary with name and model-specific parameters
        frame_avg : averaged frame to fit model function to; if none, use class attribude

        Returns
        -------
        Obtained fit parameters (first variable) and resulting PSF-fit to model (second variable).
        '''
        if modelfun is None:
            modelfun = self.psf_model
        if frame_avg is None:
            frame_avg = self.frame_avg
        
        # analogous to fit_single_frames(), but now applied to averaged, presumably drift-corrected image
        coord = np.meshgrid(np.arange(0, self.px_size_img*frame_avg.shape[0], self.px_size_img),
                            np.arange(0, self.px_size_img*frame_avg.shape[1], self.px_size_img))
        coord = np.vstack([coord[0].flatten(), coord[1].flatten()])
        coord_center = np.mean(coord, axis=1)
        
        if modelfun['name'] == 'polynomial':
            f_fit = lambda xy, x0, y0, *p: self._f_polynomial(xy, x0, y0, p, k=modelfun['k'])
            p_init = np.zeros((modelfun['k'] + 1, modelfun['k'] + 1))
            px_center = np.rint(coord_center/self.px_size_img).astype(int)
            p_init[1,0] = (np.mean(frame_avg[[px_center[0]-1, px_center[0]+1], px_center[1]]) - frame_avg[px_center[0], px_center[1]])/self.px_size_img
            p_init[0,1] = (np.mean(frame_avg[px_center[0], [px_center[1]-1, px_center[1]+1]]) - frame_avg[px_center[0], px_center[1]])/self.px_size_img
            p0 = [coord_center[0], coord_center[1], *p_init.flatten()]
        elif modelfun['name'] == 'doughnut':
            f_fit = self._f_doughnut
            p0 = [coord_center[0], coord_center[1], 350, np.mean(frame_avg)*frame_avg.size]
        elif modelfun['name'] == 'gauss':
            f_fit = self._f_gauss
            p0 = [coord_center[0], coord_center[1], 350, np.mean(frame_avg)*frame_avg.size]
        else:
            raise ValueError("modelfun['name'] must be 'polynomial', 'doughnut' or 'gauss'.") 
          
        fit_res, cov = curve_fit(f_fit, coord, frame_avg.flatten(), p0=p0)  # perform non-linear least squares fit
        
        self.f_fit = f_fit
        self.p_fit = fit_res[2:]    # exclude x0 and y0 (coordinates of center position)
        
        return fit_res, f_fit(coord, *fit_res).reshape(frame_avg.shape)


    def set_psf(self, grid_size, px_size_grid=None, p_fit=None):   
        '''
        Generate centered PSF-grid (using fit obtained by psf_fit()) at specified grid size and pixel size.

        Parameters
        ----------
        grid_size : size of coordinate grid per axis in nm; currently only square grids supported
        px_size_grid : pixel size in nm at which to calculate PSF-grid, or use same as images used for calibration
        p_fit : Specify fit parameters, or use previously obtained ones; exclude center coordinates
        
        Returns
        -------
        PSF according to fit function at spec; ensure only positive values
        '''
        if px_size_grid is None:
            px_size_grid = self.px_size_img
        if p_fit is None:
            p_fit = self.p_fit
        
        coord_grid = np.meshgrid(np.arange(0, grid_size, px_size_grid), np.arange(0, grid_size, px_size_grid))      # create coordinate mesh
        coord = np.vstack([coord_grid[0].flatten(), coord_grid[1].flatten()])   # format for compatibility with f_fit
        coord_center = np.mean(coord, axis=1)
        
        f_psf = self.f_fit(coord, *coord_center, *p_fit)
        psf = f_psf.reshape(coord_grid[0].shape)    # reshape flattened 
        
        psf = np.where(psf > 1e-32, psf, np.zeros(psf.shape) + 1e-32)   # don't want <=0 when forming log for MLE
        self.psf = psf
        self.px_size = px_size_grid
        self.grid_size = grid_size
        
        return psf


    def interpolate_psf_image(self, grid_size, px_size_grid=None, img_psf=None):  
        '''
        Directly interpolate PSF-grid from image without fitting a model function. Crop symmetrically around image center.

        Parameters
        ----------
        grid_size : size of coordinate grid per axis in nm; should be smaller than image size; currently only square grids supported
        px_size_grid : pixel size in nm at which to calculate PSF-grid, or use same as images used for calibration
        img_psf : image of PSF; if none, use averaged frames

        Returns
        -------
        Interpolated PSF-grid.
        '''
        if px_size_grid is None:
            px_size_grid = self.px_size_grid
        if img_psf is None:
            img_psf = self.frame_avg
            
        img_psf = np.array(img_psf)
        n_px_img = np.array(img_psf.shape)
        img_size = n_px_img*self.px_size_img
        d_size = img_size - grid_size     #amount by which psf image is larger (or smaller) than grid
        if d_size[0] < px_size_grid or d_size[1] < px_size_grid:
            warnings.warn('PSF-image is smaller than grid! This might lead to artifacts for MLE localization!')
    
        # set image coordinate mesh
        img_px_x, img_px_y = np.meshgrid(np.arange(0, img_size[0], self.px_size_img) + self.px_size_img/2, 
                                         np.arange(0, img_size[1], self.px_size_img) + self.px_size_img/2)
        # set grid coordinate mesh
        grid_x, grid_y = np.meshgrid(np.arange(0, grid_size, px_size_grid) + d_size[0]/2,
                                     np.arange(0, grid_size, px_size_grid) + d_size[1]/2)
        
        # interpolate psf image at grid points
        psf = griddata((img_px_x.flatten(), img_px_y.flatten()), img_psf.flatten(), (grid_x, grid_y), method='cubic')
        psf = np.nan_to_num(psf)    #to handle cases where image is smaller than grid
        psf = np.where(psf > 1e-32, psf, np.zeros(psf.shape) + 1e-32)   # don't want <=0 when forming log for MLE
        self.psf = psf
        self.px_size = px_size_grid
        self.grid_size = grid_size
        
        return psf


        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    #%% example psf calibration workflow
    n_px = 512      # numer of pixels per axis
    psfdata = data_raw = np.fromfile('minflux_data\\20220422_0101_psfCalibration.rcstk', dtype=np.int16).reshape(-1, n_px, n_px)    # read raw data 
    psfdata = psfdata[:, 180:210, 390:420]          # crop to isolate single bead
    psf = PSFcalibration(psfdata, 27)       # initialize psf-calibration with 27 nm pixel size

    # fit single model function to single frames to extract center coordinates
    # psf.fit_single_frames(modelfun={'name': 'polynomial', 'k': 2})
    psf.fit_single_frames(modelfun={'name': 'doughnut'})
    # correct drifts between frames
    psf.correct_drifts(k_smooth=2, roi=[500, 500])
    # fit PSF to obtain model
    # fitresult, fitdata = psf.psf_fit(modelfun={'name': 'doughnut'})
    fitresult, fitdata = psf.psf_fit(modelfun={'name': 'polynomial', 'k': 9})
    
    # psf_calibrated_data = psf.set_psf(600, 10)

    #%% visualize results of psf-calibration
    fig, ax = plt.subplots(1, 2, num=30, clear=True)
    max_all = np.max([psf.frame_avg, fitdata])
    imgaxdata = ax[0].imshow(psf.frame_avg, vmin=0, vmax=0.9*max_all, extent=(0, psf.frame_avg.shape[0]*psf.px_size_img, 0, psf.frame_avg.shape[1]*psf.px_size_img), cmap='cividis')
    imgaxfit = ax[1].imshow(fitdata, vmin=0, vmax=0.9*max_all, extent=(0, fitdata.shape[0]*psf.px_size_img, 0, fitdata.shape[1]*psf.px_size_img), cmap='cividis')
    
    divider = make_axes_locatable(ax[0])   # this and next line needed for colorbar to match plot height
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(imgaxdata, cax=cax)
    
    divider = make_axes_locatable(ax[1])   # this and next line needed for colorbar to match plot height
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(imgaxfit, cax=cax)
    
    ax[0].set_title('data (avg.)')
    ax[1].set_title('fit')
    
    
    plt.tight_layout()
    
    fig.savefig('minflux_data\\psf_calibration_fit.pdf')

    fig, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            ax[i,j].imshow(psfdata[5*i+j])
    