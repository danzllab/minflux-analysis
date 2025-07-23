# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:52:47 2023

@author: jvorlauf_admin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import lombscargle  #Lomb-Scargle periodogram for spectral analysis of unevenly sampled signals

import warnings


class MinfluxAnalysisBeadSample:
    
    def __init__(self, parameters, data, localize, t_crop=None, save_plots=False):
        
        self.parameters = parameters
        self.data = data
        self.localize = localize
        if t_crop:
            pass
        self.save_plots = save_plots
        
        self.plot_ind = 20
    
    
    def _subplots_template(self, nrows=1, ncols=1, width=6, sharex=False, sharey=False):
        
        fig, ax = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey,
                               figsize=(width, 4), dpi=200, 
                               num=self.plot_ind, clear=True)
        self.plot_ind += 1
        return fig, ax
    
        
    def fft_counts(self, counts=None, average_exp=True, sum_counts=True, f_lim=None, 
                   log_x=False, log_y=True):
        if counts is None:
            counts = self.localize.counts_thresh
        
        if average_exp:     #avoid interpolation by cropping all exp to length of shortest
            len_min = np.min([c.shape[0] for c in counts])
            len_min = 5000
            counts = [c[:len_min] for c in counts]    
        if sum_counts: 
            counts_sum = self.localize.sum_counts(counts)
            
        df = 1/(self.parameters.n_tcp * self.parameters.t_tcp)
        
        fft_exp = []
        f_exp = []
        for i in range(len(counts)):
            
            n_below_thresh = np.count_nonzero(self.localize.count_mask[i] == False)
            if n_below_thresh > 0:
                warnings.warn(f'In exp {i}, {n_below_thresh} cycles have counts below the threshold. Might lead to artifacts in fft!')
            
            if sum_counts:
                fft_data = np.abs(np.fft.rfft(counts_sum[i], axis=0))
            else:
                # t_on = self.data.t_exp[i][self.data.count_mask[i]]
                # t_on = t_on[:len_min]
                # f_data = np.linspace(0.1, df/2, counts[i].shape[0])
                # ft_cols = []
                # for counts_col in counts[i].transpose():
                #     # ft_cols.append(lombscargle(t_on, counts_col, f_data, normalize=True))
                #     ft_cols.append(np.abs(self._nudft(t_on, counts_col, f_data)))
                # fft_data = np.vstack(ft_cols).transpose()
                fft_data = np.abs(np.fft.rfft(counts[i], axis=0))
            f_data = df*np.fft.rfftfreq(counts[i].shape[0])
            if f_lim:
                if f_lim[1] is None:
                    f_lim[1] =  1.01*max(f_data)
                f_mask = np.logical_and(f_data > f_lim[0], f_data < f_lim[1])
                f_data = f_data[f_mask]
                fft_data = fft_data[f_mask]
            fft_exp.append(fft_data)
            f_exp.append(f_data)
            
        if average_exp:
            fft_exp = [np.average([fft for fft in fft_exp], axis=0)]
            
        if sum_counts:
            fig, ax = self._subplots_template(1, 1)
            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')
             
            ax.set_xlabel('f [Hz]')
            if f_lim: 
                ax.set_xlim(*f_lim)
            
            for f, fft in zip(f_exp, fft_exp):
                ax.plot(f, fft)
                
        else:
            fig, ax = self._subplots_template(self.parameters.n_tcp, 1, 
                                              sharex=True, sharey=True)
            for i in range(self.parameters.n_tcp):
                if log_x:
                    ax[i].set_xscale('log')
                if log_y:
                    ax[i].set_yscale('log')
                if f_lim: 
                    ax[i].set_xlim(*f_lim)
                
                for f, fft in zip(f_exp, fft_exp):
                    ax[i].plot(f, fft[:,i])
          
            ax[-1].set_xlabel('f [Hz]')
        plt.tight_layout()
        if self.save_plots: plt.savefig(self.parameters.file_name + '_fftCounts.png')       
     

    def fft_localizations(self, localizations=None, average_dims=False, f_lim=None, 
                          log_x=False, log_y=True):
          
        if localizations is None:
            localizations = self.data.localizations
            
        df = 1/(self.parameters.n_tcp * self.parameters.t_tcp)
             
        fft_exp = []
        f_exp = []
        for i in range(len(localizations)):
            
            n_below_thresh = np.count_nonzero(self.localize.count_mask[i] == False)
            if n_below_thresh > 0:
                warnings.warn(f'In exp {i}, {n_below_thresh} cycles have counts below the threshold. Might lead to artifacts in fft!')
             
            
         
        
    def precision_series(self, n_photon_steps, counts=None, estimator='lms',
                         average_exp=False, exponent=None, plot_localization_series=False):
        if counts is None:
            counts = self.localize.counts_raw
        n_photon_series = np.arange(*n_photon_steps)
        
        if average_exp:
            std = np.zeros((n_photon_series.size, ))
            stderr = np.zeros((n_photon_series.size, ))
        else:
            std = np.zeros((n_photon_series.size, len(counts)))
            stderr = np.zeros((n_photon_series.size, len(counts)))
            
        if plot_localization_series:
            n_rows = np.sqrt(len(n_photon_series))
            n_cols = n_rows if n_rows % 1 == 0 else n_rows + 1
            n_rows, n_cols = int(n_rows), int(n_cols)
            fig, ax = self._subplots_template(n_rows, n_cols, sharex=True, sharey=True)
            
        for i in range(n_photon_series.size):
            events_binned = self.localize.bin_emission_events(counts=counts, n_bin=n_photon_series[i])
            localizations = self.localize.estimate_position(counts_thresh=events_binned, estimator=estimator)
            
            std_exp = np.zeros(len(localizations))
            stderr_exp = np.zeros(len(localizations))
            for j in range(len(localizations)):
                std_xy = np.std(localizations[j], axis=0)
                stderr_xy = std_xy/np.sqrt(localizations[j].shape[0])
                std_exp[j] = np.mean(std_xy)
                stderr_exp[j] = np.mean(stderr_xy)
                
                if plot_localization_series:
                    if localizations[j].size > 0:
                        ax[i//n_cols, i%n_cols].scatter(localizations[j][:,0], localizations[j][:,1], zorder=2,
                                                        facecolors='none', edgecolors='C'+str(j), alpha = 0.5, s=5, linewidth=0.5)
                        ax[i//n_cols, i%n_cols].set_aspect('equal')
                        ax[i//n_cols, i%n_cols].set_title(f'N = {n_photon_series[i]}')
                        plt.tight_layout()
            
            if average_exp:
                std[i] = np.mean(std_exp)
                stderr[i] = np.mean(stderr_exp)
            else:
                std[i] = std_exp
                stderr[i] = stderr_exp
            
        if exponent:
            fit_fun = lambda N, p_scale, p_offset: \
                p_scale * self.parameters.L/(N**exponent) + p_offset
        else:
            fit_fun = lambda N, p_scale, p_offset, p_exp: \
                p_scale * self.parameters.L/(N**p_exp) + p_offset
            
        fig, ax = self._subplots_template()
        if average_exp:
            std_fit, _ = curve_fit(fit_fun, n_photon_series, std, sigma=stderr)
            ax.errorbar(n_photon_series, std, yerr=stderr, fmt='o', capsize=5,
                        markerfacecolor='none', linestyle='none')
            ax.plot(n_photon_series, fit_fun(n_photon_series, *std_fit), color='C0')
            
        else:
            std_fit = []
            for i in range(len(counts)):
                std_fit_exp, _ = curve_fit(fit_fun, n_photon_series, std[:,i], sigma=stderr[:,i])
                ax.errorbar(n_photon_series, std[:,i], yerr=stderr[:,i], fmt='o', capsize=5,
                            markerfacecolor='none', markeredgecolor='C'+str(i), linestyle='none')
                ax.plot(n_photon_series, fit_fun(n_photon_series, *std_fit_exp), color='C'+str(i))
                std_fit.append(std_fit_exp)
            std_fit = np.mean(std_fit, axis=0)
        
        print('\nprecision series fit result:')
        print('scale factor: ', std_fit[0])
        print('offset: ', std_fit[1])
        if not exponent: print('exponent: ', std_fit[2])
        print('\n')

        if self.save_plots: plt.savefig(self.parameters.file_name + '_precisionSeries.png')      
        
        
        
        
    def _nudft(self, t, y, f):     # non-uniform discrete Fourier Transform Type 2
        # slower than for-loop
        # f_trans = f[np.newaxis,:].transpose()
        # ft_arr = y*np.exp(-2*np.pi*1j*t*f_trans)
        # ft = np.sum(ft_arr, axis=1)

        ft = np.zeros(f.shape, dtype=np.cdouble)
        for k in  range(len(f)):
            ft_coeff = y*np.exp(-2*np.pi*1j*t*f[k])
            ft[k] = np.trapz(ft_coeff, f)
        
        # ft = _nudft_cython(t, y, f)
           
        return ft



class MinfluxAnalysisBlinkingSample:
    def __init__(self, data):
        self.data = data


# TODO compute auto-correlation of localization traces (potentially for only 1 data point per ON-event - mfxloc.assign_emission_events)

