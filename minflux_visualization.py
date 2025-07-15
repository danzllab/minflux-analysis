# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:52:13 2023

@author: jvorlauf_admin
"""

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
                 plot_style='default', plot_interface='inline'):
       
        self.parameters = parameters
        self.data = data
        plt.style.use(plot_style)
        self.save_plots = save_plots
        if save_plots == '.pdf': mpl.style.use('matplotlibrc_adapted_pdf')
        self.plot_ind = 0   # got tired of manually numbering plots, but want the same number when running the script multiple times
        

    def _f_gauss(self, fwhm=None, center=None, intensity=1, grid_size=None, px_size=None):
        if fwhm is None: 
            fwhm = self.shape_param
        if grid_size is None: 
            try:
                grid_size = self.data.grid_size
            except AttributeError:
                grid_size = 2*self.data.L
        if px_size is None:
            try:
                px_size = self.data.px_size_grid
            except AttributeError:
                px_size = 1
        if center is None: 
            center = [grid_size/2, grid_size/2]
        
        coord = np.arange(0, grid_size, px_size)
        xx, yy = np.meshgrid(coord-center[0], coord-center[1])  #create grids corresponding to x and y coordinates
        rr2 = xx**2 + yy**2
        sigma = fwhm/(2*np.sqrt(2*np.log(2)))
        pixel_intensities = intensity/(sigma*np.sqrt(2*np.pi))*np.exp(-rr2/(2*sigma**2))
        
        return pixel_intensities
    
    
    def _subplots_template(self, nrows=1, ncols=1, width=6):
        fig, ax = plt.subplots(nrows, ncols, figsize=(width, 4), dpi=200, num=self.plot_ind, clear=True)
        self.plot_ind += 1
        return fig, ax
    
    
    def plot_localizations_scatter(self, localizations=None, show_lines=False, color_code='time'):
        if localizations is None:
            localizations = self.data.localizations
         
        fig, ax = self._subplots_template()
        _draw_localizations_scatter(fig, ax, localizations, self.data.t_mask, show_lines=show_lines, color_code=color_code)

        ax.set_xlabel('x [nm]')
        ax.set_ylabel('y [nm]')
        
        plt.tight_layout()
        if self.save_plots: plt.savefig(self.parameters.file_name + '_localizations_' + color_code + 'Coding' + self.save_plots)
        plt.show()
          

    def plot_localizations_histogram(self, localizations=None, px_size=1, vlim=None,
                                    shift_hist=False):
        if localizations is None:
            localizations = self.data.localizations
        loc_all = np.vstack(localizations)
        
        n_bins = [round((max(loc_all[:,0]) - min(loc_all[:,0]))/px_size),
                  round((max(loc_all[:,1]) - min(loc_all[:,1]))/px_size)]
        
        # define range to get rid of rounding error
        xyrange = [[min(loc_all[:,0]), min(loc_all[:,0]) + (n_bins[0] - 2)*px_size],
                   [min(loc_all[:,1]), min(loc_all[:,1]) + (n_bins[1] - 2)*px_size]]
        
        fig, ax = self._subplots_template()
        ax.set_aspect('equal')
        
        h = ax.hist2d(loc_all[:,0], loc_all[:,1], n_bins, range=xyrange, cmap='inferno')
        h_img = h[3]
        # currently hist is always shifted by 1 to left and right, could adapt for more px?
        if shift_hist is True:
            h_shift = h[0][1:-1,1:-1] + 1/4*(h[0][0:-2,1:-1] + h[0][2:,1:-1] + h[0][1:-1,0:-2] + h[0][1:-1,2:])
            h_shift *= 0.5
            ax.clear()
            h_img = ax.imshow(h_shift.transpose()[::-1,:], extent=[h[1][0], h[1][-1], h[2][0], h[2][-1]],
                             cmap='inferno')
        
        if vlim is not None:
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
        if self.save_plots: plt.savefig(self.parameters.file_name + '_locHist' + self.save_plots)
    
    
    def plot_localizations_gauss(self, localizations=None, px_size=0.1, 
                                    sigma=None, vlim=None):
        if localizations is None:
            localizations = self.data.localizations
        loc_all = np.vstack(localizations)
        if sigma is None:
            sigma = 1
            # counts_loc = 0
            # TODO calculate uncertainty from CRB?
        fwhm = 2.355 * sigma
        
        n_bins = [round((max(loc_all[:,0]) - min(loc_all[:,0]))/px_size),
                  round((max(loc_all[:,1]) - min(loc_all[:,1]))/px_size)]
        
        # define range to get rid of rounding error
        xyrange = [[min(loc_all[:,0]), min(loc_all[:,0]) + (n_bins[0] - 2)*px_size],
                   [min(loc_all[:,1]), min(loc_all[:,1]) + (n_bins[1] - 2)*px_size]]
        
        fig, ax = self._subplots_template()
        ax.set_aspect('equal')
        
        if type(fwhm) == float:
            h = ax.hist2d(loc_all[:,0], loc_all[:,1], n_bins, range=xyrange, cmap='grey')
            h_img = h[3]
            
            h_gauss = fftconvolve(h[0], self._f_gauss(fwhm=fwhm, intensity=1, grid_size=5*fwhm, px_size=px_size), mode='same')
            ax.clear()
            h_img = ax.imshow(h_gauss.transpose()[::-1,:], extent=[h[1][0], h[1][-1], h[2][0], h[2][-1]],
                             cmap='grey')
            
        else:
            h = np.zeros(n_bins)
            for loc, fw in zip(loc_all, fwhm):
                h += self._f_gauss(fwhm=fw, center=loc, intensity=1 , n_grid=n_bins)
        
        if vlim is not None:
            h_img.set_clim(vmin=vlim[0], vmax=vlim[1])
            
        divider = make_axes_locatable(ax)   # this and next line needed for colorbar to match plot height
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(h_img, cax=cax)
        
        ax.set_xlabel('x [nm]')
        ax.set_ylabel('y [nm]')
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        plt.tight_layout()
        if self.save_plots: plt.savefig(self.parameters.file_name + '_locGauss' + self.save_plots)
        plt.show()
        
        
    def plot_localization_traces(self, localizations=None, p_drift=None, centering=False):
        if localizations is None:
            localizations = self.data.localizations
        # loc_all = np.vstack(localizations)
        # t = np.hstack(self.data.t_mask)
        # t = t - max(t)
        t, loc_all = _sort_time_coded_data(self.data.t_mask, localizations)
        
        if centering:
            loc_mean = np.mean(loc_all, axis=0)
            loc_all -= loc_mean
       
        fig, ax = self._subplots_template()
        ax.set_ylabel('localizations [nm]')
        ax.plot(t, loc_all, alpha=0.6, linewidth=0.5)
        ax.set_xlabel('time [s]')
         
        if p_drift:
            drifts = []
            avg_drifts = (len(localizations) != len(p_drift))   # True if no separate drift function is passed for every exp
            for i in range(len(localizations)):
                j = 0 if avg_drifts else i      
                if p_drift[j].size > 0:
                    t_fit = np.linspace(0, 1, len(localizations[i]))
                    k_fit = len(p_drift[j]) - 1
                    drift_x = np.sum([p_drift[j][l,0]*t_fit**(k_fit - l) for l in range(k_fit + 1)], axis=0)
                    drift_y = np.sum([p_drift[j][l,1]*t_fit**(k_fit - l) for l in range(k_fit + 1)], axis=0)
                    drift_exp = np.vstack([drift_x, drift_y]).transpose()
                    drifts.append(drift_exp)
                # drift_exp -= p_drift[i][-1]
        
            # drifts_all = np.vstack(drifts)
            t, drifts_all = _sort_time_coded_data(self.data.t_mask, drifts)
            ax.plot(t, drifts_all[:,0], color='C0', linewidth = 2)
            ax.plot(t, drifts_all[:,1], color='C1', linewidth = 2)
        
        plt.tight_layout()
        if self.save_plots: plt.savefig(self.parameters.file_name + '_locTraces' + self.save_plots)        
        plt.show()
        
    
    def plot_count_traces(self, counts=None, show_trace_segmentation=True):
        if counts is None:
            counts = self.data.counts_processed
        # counts_all = np.vstack(counts)
        # counts_sum = np.hstack(_sum_counts(counts))
            
        t, counts_all = _sort_time_coded_data(self.data.t_exp, counts)
        
        fig, ax = self._subplots_template()
        plt.xlabel('time [s]')
        plt.ylabel('photon counts')
        plt.plot(t, counts_all, linewidth=0.5)
        plt.ylim(bottom=min(np.min(counts_all), 0))     #to deal with negative BG counts after BG subtraction
            
        if show_trace_segmentation: 
            _, counts_sum = _sort_time_coded_data(self.data.t_exp, _sum_counts(counts))
            _, count_mask_sorted = _sort_time_coded_data(self.data.t_exp, self.data.count_mask)
            plt.plot(t, counts_sum/self.parameters.n_tcp, linestyle=':', linewidth=0.3)
            plt.fill_between(t, counts_sum/self.parameters.n_tcp, min(0, np.min(counts_all)), where=np.hstack(count_mask_sorted), 
                             color='C4', alpha=0.5)
            # plt.fill_between(t, counts_sum/n_coord, where=np.hstack(self.background_mask), 
            #                   color='C5', alpha=0.5)
        
        if self.save_plots: plt.savefig(self.parameters.file_name + '_countTraces' + self.save_plots)
        plt.show()
        
        
    def plot_count_histogram(self, counts=None, n_bins=None, d_bins=1, log=False):
        if counts is None:
            counts = self.data.counts_processed
        counts_sum = np.hstack(_sum_counts(counts))
        if n_bins is None:
            bins = np.arange(0, int(max(counts_sum)) + 1, d_bins)
        else:
            bins = np.arange(0, n_bins + 1, d_bins)
        
        fig, ax = self._subplots_template()
        hist_data, _, _ = plt.hist(counts_sum, bins=bins, log=log)
        plt.xlabel('counts')
        plt.ylabel("number of occurences")
         
        poisson_mix = lambda k, a0, mu0, a1, mu1: a0*poisson.pmf(k, mu0) + \
                                                a1*poisson.pmf(k, mu1)
                    
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
        
        if self.save_plots: plt.savefig(self.parameters.file_name + '_countHist' + self.save_plots)
        plt.show()



