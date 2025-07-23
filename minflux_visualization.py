# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:52:13 2023

@author: jvorlauf_admin
"""

import warnings

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import poisson

import matplotlib.pyplot as plt
import matplotlib as mpl
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')  # inserts thousands separator
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

from minflux_utils import _sort_time_coded_data, _sum_counts, \
    _draw_localizations_scatter


class MinfluxVisualization2D:
    
    def __init__(self, parameters, data, save_plots=False,
                 plot_style='default'):
       
        self.parameters = parameters
        self.data = data
        plt.style.use(plot_style)
        self.save_plots = save_plots
        if save_plots == '.pdf': mpl.style.use('matplotlibrc_adapted_pdf')
        self.plot_ind = 0   # want the same figure numbers when running the script multiple times
        

    def _f_gauss(self, fwhm=None, center=None, intensity=1, grid_size=None, px_size=None):
        '''2D Gaussian function.'''
        if fwhm is None: 
            fwhm = self.shape_param
        if grid_size is None: 
            try:
                grid_size = self.data.grid_size
            except AttributeError:
                warnings.warn('Not grid size given. Set to 2 * L.')
                grid_size = 2*self.data.L
        if px_size is None:
            try:
                px_size = self.data.px_size_grid
            except AttributeError:
                warnings.warn('No pixel size given. Set to 1 nm.')
                px_size = 1
        if center is None: 
            center = [grid_size/2, grid_size/2]
        
        if hasattr(grid_size, '__iter__'):      # create rectangular coordinate grid 
           # coord = [np.arange(-grid_size[0]/2, grid_size[0]/2, px_size), np.arange(-grid_size[1]/2, grid_size[1]/2, px_size)]
           coord = [np.linspace(0, gs, round(gs/px_size)) for gs in grid_size]
        else:       # create square coordinate grid 
            coord = [np.linspace(0, grid_size, round(grid_size/px_size)), ]*2
        xx, yy = np.meshgrid(coord[1] - center[1], coord[0] - center[0])  #create grids corresponding to x and y coordinates
        rr2 = xx**2 + yy**2
        sigma = fwhm/(2*np.sqrt(2*np.log(2)))
        pixel_intensities = intensity/(sigma*np.sqrt(2*np.pi))*np.exp(-rr2/(2*sigma**2))
        
        return pixel_intensities
    
    
    def _subplots_template(self, nrows=1, ncols=1, width=6):
        fig, ax = plt.subplots(nrows, ncols, figsize=(width, 4), dpi=200, num=self.plot_ind, clear=True)
        self.plot_ind += 1
        return fig, ax
    
    
    def plot_localizations_scatter(self, localizations=None, show_lines=False, color_code='time'):
        '''
        Visualize localizations as scatter plot. Extent deduced from localizations.

        Parameters
        ----------
        localizations : specify or use the ones saved in minflux data
        show_lines : boolean indicating whether to display lines connecting subsequent localizations
        color_code : color-code according to localization time ("time") or experiment/tile index ("tile")

        '''
        if localizations is None:
            localizations = self.data.localizations
         
        fig, ax = self._subplots_template()
        _draw_localizations_scatter(fig, ax, localizations, self.data.t_mask, show_lines=show_lines, color_code=color_code)     # create from utility function

        ax.set_xlabel('x [nm]')
        ax.set_ylabel('y [nm]')
        
        plt.tight_layout()
        if self.save_plots: plt.savefig(self.parameters.file_name + '_loc-scatter_' + color_code + '-coding' + self.save_plots, transparent=True)
        plt.show()
          

    def plot_localizations_histogram(self, localizations=None, px_size=1, vlim=None,
                                    shift_hist=False):
        '''
        Visualize localizations as 2D histogram. Extent deduced from localizations.

        Parameters
        ----------
        localizations : specify or use the ones saved in minflux data
        px_size : pixel size of output image in nm
        vlim : Lower and upper limit of value range displayed in color map; if none, show full range
        shift_hist : boolean indicating whether to overlay histogram with versions shifted by +-1 pixel to generate smoother visualization
        '''
        if localizations is None:
            localizations = self.data.localizations
        loc_all = np.vstack(localizations)
        
        n_bins = [round((max(loc_all[:,0]) - min(loc_all[:,0]))/px_size),
                  round((max(loc_all[:,1]) - min(loc_all[:,1]))/px_size)]
        xyrange = [[min(loc_all[:,0]), min(loc_all[:,0]) + (n_bins[0] - 2)*px_size],
                   [min(loc_all[:,1]), min(loc_all[:,1]) + (n_bins[1] - 2)*px_size]]
        
        fig, ax = self._subplots_template()
        ax.set_aspect('equal')
        
        h = ax.hist2d(loc_all[:,0], loc_all[:,1], n_bins, range=xyrange, cmap='inferno')    # hist2d returns histogram array, xedges, yedges, image
        h_img = h[3]
        if shift_hist: # currently hist is always shifted by 1 to left and right
            h_shift = h[0][1:-1,1:-1] + 1/4*(h[0][0:-2,1:-1] + h[0][2:,1:-1] + h[0][1:-1,0:-2] + h[0][1:-1,2:])
            h_shift *= 0.5  # to preserve overall intensity range
            ax.clear()      # delete previously plotted histogram and create new
            h_img = ax.imshow(h_shift.transpose()[::-1,:], extent=[h[1][0], h[1][-1], h[2][0], h[2][-1]],
                             cmap='inferno')
        
        if vlim is not None:    # set limits of color map
            h_img.set_clim(vmin=vlim[0], vmax=vlim[1])
            
        divider = make_axes_locatable(ax)   # this and next line needed for colorbar to match plot height
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(h_img, cax=cax)
        
        ax.set_xlabel('x [nm]')
        ax.set_ylabel('y [nm]')
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        plt.tight_layout()
        plt.show()
        if self.save_plots: plt.savefig(self.parameters.file_name + '_loc-hist' + self.save_plots, transparent=True)
    
    
    def plot_localizations_gauss(self, localizations=None, px_size=0.1, 
                                    sigma=None, vlim=None):
        '''
        Visualize localizations as 2D-Gaussians. Extent deduced from localizations.

        Parameters
        ----------
        localizations : specify or use the ones saved in minflux data
        px_size : pixel size of output image in nm
        sigma : width of Gaussians; single value or array containing one value per localization
        vlim : Lower and upper limit of value range displayed in color map; if none, show full range
        '''
        if localizations is None:
            localizations = self.data.localizations
        loc_all = np.vstack(localizations)  #stack localizations from different experiments to create array of shape (-1, 2)
        if sigma is None:
            sigma = 1
            # TODO might calculate uncertainty from CRB?
        fwhm = 2.355 * sigma
        
        n_bins = [round((max(loc_all[:,0]) - min(loc_all[:,0]))/px_size),
                  round((max(loc_all[:,1]) - min(loc_all[:,1]))/px_size)]
        xyrange = [[min(loc_all[:,0]), min(loc_all[:,0]) + (n_bins[0] - 2)*px_size],
                   [min(loc_all[:,1]), min(loc_all[:,1]) + (n_bins[1] - 2)*px_size]]
        
        fig, ax = self._subplots_template()
        ax.set_aspect('equal')
        
        if not hasattr(fwhm, '__iter__'):     # if same width for all, computationally more efficient to convolve 2D histogram with Gaussian
            h = ax.hist2d(loc_all[:,0], loc_all[:,1], n_bins, range=xyrange, cmap='inferno')    # hist2d returns histogram array, xedges, yedges, image
            h_img = h[3]
            
            h_gauss = fftconvolve(h[0], self._f_gauss(fwhm=fwhm, intensity=1, grid_size=5*fwhm, px_size=px_size), mode='same')  #convolve histogram with Gaussian
            ax.clear()      # clear previously plotted histogram
            h_img = ax.imshow(h_gauss.transpose()[::-1,:], extent=[h[1][0], h[1][-1], h[2][0], h[2][-1]],   # plot Gauss visualization; reshape for correct orientation; extent preserves spatial scale (in units of nm)
                             cmap='inferno')
            
        else:       # if different widths for localizations, add them one by one to image
            h = np.zeros([round((xy[1] - xy[0])/px_size) for xy in xyrange]) 
            for loc, fw in zip(loc_all, fwhm):
                h += self._f_gauss(fwhm=fw, center=[loc[i] - xyrange[i][0] for i in range(len(loc))], intensity=1 , 
                                   grid_size=[xy[1] - xy[0] for xy in xyrange], px_size=px_size)   #subtract minimum localization coordinates from center since mesh of _f_gauss() starts at 0
            h_img = ax.imshow(h.transpose()[::-1,:], extent=[each for xy in xyrange for each in xy], cmap='inferno')    # plot Gauss visualization; reshape for correct orientation; extent preserves spatial scale (in units of nm)
            
        if vlim is not None:    # set limits of color map
            h_img.set_clim(vmin=vlim[0], vmax=vlim[1])
            
        divider = make_axes_locatable(ax)   # this and next line needed for colorbar to match plot height
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(h_img, cax=cax)
        
        ax.set_xlabel('x [nm]')
        ax.set_ylabel('y [nm]')
        
        plt.tight_layout()
        if self.save_plots: plt.savefig(self.parameters.file_name + '_loc-gauss' + self.save_plots, transparent=True)
        plt.show()
        
        
    def plot_localization_traces(self, localizations=None, p_drift=None, centering=False):
        '''
        Plot localization coordinates against time.

        Parameters
        ----------
        localizations : specify or use the ones saved in minflux data
        p_drift : polynomial coefficients fit to localizations for drift correction; if given, drifts are overlaid with localizations; can be separate polynomials for experiments or common polynomial for all
        centering : if True, center localizations around 0 for both axes
        '''
        if localizations is None:
            localizations = self.data.localizations
        t, loc_all = _sort_time_coded_data(self.data.t_mask, localizations)     # to handle instances where a grid was scanned multiple times, hence time points might not be monotonously increasing between experiments
        
        if centering:
            loc_mean = np.mean(loc_all, axis=0)
            loc_all -= loc_mean
       
        fig, ax = self._subplots_template()
        ax.set_ylabel('Localizations [nm]')
        # ax.plot(t, loc_all, alpha=0.6, linewidth=0.5)         # alternative visualization
        ax.scatter(t, loc_all[:,0], alpha=0.3, s=5, linewidth=0.5, zorder=2)
        ax.scatter(t, loc_all[:,1], alpha=0.3, s=5, linewidth=0.5, zorder=2)
        ax.set_xlabel('Time [s]')
         
        if p_drift:     # if given, plot drift polynomial
            drifts = []
            drifts_averaged = (len(p_drift) == 1)   # True if no separate drift function is passed for every exp
            for i in range(len(localizations)):
                j = 0 if drifts_averaged else i     # if averaged, always use first/only polynomial coefficients; else use corresponding ones    
                if p_drift[j].size > 0:         # if drift polynomial exists for given exp, generate drift curves
                    t_fit = np.linspace(0, 1, len(localizations[i]))    #scales to (0, 1) as previously done for fitting; due to numerical instability for big values
                    k_fit = len(p_drift[j]) - 1
                    drift_x = np.sum([p_drift[j][l,0]*t_fit**(k_fit - l) for l in range(k_fit + 1)], axis=0)
                    drift_y = np.sum([p_drift[j][l,1]*t_fit**(k_fit - l) for l in range(k_fit + 1)], axis=0)
                    drift_exp = np.vstack([drift_x, drift_y]).transpose()
                    drifts.append(drift_exp)
                # drift_exp -= p_drift[i][-1]   # in case constant component should be removed
        
            t, drifts_all = _sort_time_coded_data(self.data.t_mask, drifts)     # to handle instances where a grid was scanned multiple times, hence time points might not be monotonously increasing between experiments
            ax.plot(t, drifts_all[:,0], color='C0', linewidth=1.5)
            ax.plot(t, drifts_all[:,1], color='C1', linewidth=1.5)
        
        plt.tight_layout()
        if self.save_plots: plt.savefig(self.parameters.file_name + '_localization-traces' + self.save_plots, transparent=True)        
        plt.show()
        
    
    def plot_count_traces(self, counts=None, show_trace_segmentation=True):
        '''
        Plot count traces against time.

        Parameters
        ----------
        counts : specify or use the ones saved in minflux data
        show_trace_segmentation : boolean indicating whether to overlay average counts and detected emission events
        '''
        if counts is None:
            counts = self.data.counts_processed
            
        t, counts_all = _sort_time_coded_data(self.data.t_exp, counts)  # to handle instances where a grid was scanned multiple times, hence time points might not be monotonously increasing between experiments
        
        fig, ax = self._subplots_template()
        plt.xlabel('Time [s]')
        plt.ylabel('Photon counts')
        plt.plot(t, counts_all, linewidth=0.5)
        plt.ylim(bottom=min(np.min(counts_all), 0))     #to deal with negative BG counts after BG subtraction
            
        if show_trace_segmentation: 
            _, counts_sum = _sort_time_coded_data(self.data.t_exp, _sum_counts(counts))
            _, count_mask_sorted = _sort_time_coded_data(self.data.t_exp, self.data.count_mask)
            plt.plot(t, counts_sum/self.parameters.n_tcp, linestyle=':', linewidth=0.3)     #plot average counts, so level if comparable to single traces
            plt.fill_between(t, counts_sum/self.parameters.n_tcp, min(0, np.min(counts_all)), where=np.hstack(count_mask_sorted), 
                             color=f'C{self.parameters.n_tcp}', alpha=0.5)      # fill area underneath average count trace where emission event was detected

        
        if self.save_plots: plt.savefig(self.parameters.file_name + '_count-traces' + self.save_plots, transparent=True)
        plt.show()
        
        
    def plot_count_histogram(self, counts=None, n_bins=None, d_bins=1, log=False):
        '''
        Show histogram of summed photon counts per iteration. 

        Parameters
        ----------
        counts : specify or use the ones saved in minflux data
        n_bins : specify number of bins to deduce from maximum counts and d_bins
        d_bins : distance/size of bins
        log : boolean indicating whether to plot histogram on logarithmic scale (y-axis only)

        Returns
        -------
        None.

        '''
        if counts is None:
            counts = self.data.counts_processed
        counts_sum = np.hstack(_sum_counts(counts))     # flatten across experiments
        if n_bins is None:      # if not given, create bins from 0 to maximum counts
            bins = np.arange(0, int(max(counts_sum)) + 1, d_bins)   # +1 since np.arange() ends with open interval
        else:
            bins = np.arange(0, n_bins + 1, d_bins)
        
        fig, ax = self._subplots_template()
        hist_data, _, _ = plt.hist(counts_sum, bins=bins, log=log)
        plt.xlabel('Counts')
        plt.ylabel('Number of occurences')
         
        
        ## if count levels exhibit sufficiently narrow distribution, fitting a mix of 3 Poisson distributions could automatically extract levels of background, single emitter, and multiple emitters
        # poisson_mix = lambda k, a0, mu0, a1, mu1: a0*poisson.pmf(k, mu0) + \
        #                                         a1*poisson.pmf(k, mu1)
                    
        # p_fit, _ = curve_fit(poisson_mix, bins[:-1], hist_data, p0=[50000, 30, 10000, 60])
        # print(p_fit)
        
        # plt.plot(bins[:-1], poisson_mix(bins[:-1], *p_fit), label='Poisson mix')


        # poisson_offset = lambda k, a0, mu0, offset: a0*poisson.pmf(k, mu0, round(offset))
                    
        # p_fit, _ = curve_fit(poisson_offset, bins[:-1], hist_data, p0=[3000, 300, 250])
        # print(p_fit)
        
        # plt.plot(bins[:-1], poisson_offset(bins[:-1], *p_fit), label='single Poisson (250 offset)')
        
        
        # gauss = lambda x, a, mu, s: a*np.exp(-(x - mu)**2/2/s**2)
                    
        # p_fit, _ = curve_fit(gauss, bins[:-1], hist_data, p0=[500, 2000, 500])
        # print(p_fit)
        
        # plt.plot(bins[:-1], gauss(bins[:-1], *p_fit), label='Gauss')
        
        # plt.legend()
        
        if self.save_plots: plt.savefig(self.parameters.file_name + '_count-hist' + self.save_plots, transparent=True)
        plt.show()



