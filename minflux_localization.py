# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:52:03 2023

@author: jvorlauf_admin
"""

import warnings
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt


class MinfluxLocalization2D:
    def __init__(self, parameters, data, psf_data=None):
         
        if hasattr(data, 'counts_tiles'):
            self.counts_raw = data.counts_tiles
            self.counts_processed = data.counts_tiles
        else:
            self.counts_raw = [data.counts_raw]
            self.counts_processed = [data.counts_raw]
        if not hasattr(data, 't_exp'):
            warnings.warn('No timestamps found. Create with t = 1 sec.')
            data.create_timestamps(1)
        
        self.parameters = parameters
        self.data = data
        self.psf_data = psf_data
        
    
    def remove_activation_counts(self, counts=None):
        if counts is None:
            counts = self.counts_processed
        
        counts_filtered = []
        for counts_exp in counts:
            act_cycle = np.count_nonzero(counts_exp, axis=1) <= 1    #convention of dcs program: during activation first exposure = sum of counts, others 0
            counts_filtered_exp = np.where(np.vstack((act_cycle, )*self.parameters.n_tcp).transpose(), 
                                           np.transpose((counts_exp[:,0]/self.parameters.n_tcp, )*self.parameters.n_tcp),   # set avg. counts for whole TCP
                                           counts_exp)
            counts_filtered.append(counts_filtered_exp)
        
        self.counts_processed = counts_filtered
        return counts_filtered
 
    
    def bin_counts(self, counts=None, k_bin=5):
        '''bin counts  of multiple exposures to improve precision; bin k_bin data points together'''
        if counts is None:
            counts = self.counts_processed 
        
        counts_binned = []
        t_exp = [t[::k_bin] for t in self.data.t_exp]
        for i, counts_exp in enumerate(counts):
            c_dim = counts_exp.shape
            stub = c_dim[0]%k_bin
            if stub: 
                t_exp[i] = t_exp[i][:-1]
                counts_exp = counts_exp[:-stub]     #crop last entries to have number of cycles fit k_bin
            counts_exp_binned = counts_exp.reshape(-1, k_bin, self.parameters.n_tcp).sum(axis=1)
            counts_binned.append(counts_exp_binned)
            
        self.counts_processed = counts_binned
        
        self.parameters.t_tcp *= k_bin
        self.data.t_exp = t_exp
        
        return counts_binned

        
    def filter_counts(self, filter_type, t_filter, counts=None, normalize=True):
        ''''filter_weights: 1D array containing the weights of the filter'''
        if counts is None:
            counts = self.counts_processed
        
        if filter_type == 'gauss': 
            filter_weights = np.exp(-np.arange(-250, 250)**2/(2*t_filter**2))
        elif filter_type == 'box': 
            filter_weights = np.ones(t_filter)
        elif filter_type == 'triangle': 
            filter_weights = np.hstack((np.arange(t_filter), np.arange(t_filter, 0, -1)))
        else:
            raise ValueError("'filter_type' must be 'gauss', 'box' or 'triangle'.")
        
        if normalize: filter_weights = filter_weights/sum(filter_weights)
        filter_weights = filter_weights.reshape(-1, 1)  #reshape f_filter to convolve along axis 0
        counts_filtered = [fftconvolve(counts_exp, filter_weights, mode='same') for counts_exp in counts]   #mode "same" gives edge effects but no phase; accept for now and reevaluate as needed
        
        self.counts_processed = counts_filtered
        return counts_filtered
    
    
    def sum_counts(self, counts=None):
        if counts is None:
            counts = self.counts_processed          
        counts_sum = [np.sum(counts_exp, axis=1) for counts_exp in counts]
        return counts_sum
    
    
    def threshold_counts(self, threshold, counts=None, cfr_max=0.25, t_min=1, 
                         bin_var=None, thresh_var=0.1, subtract_background=True):
        if counts is None:
            counts = self.counts_processed          
        counts_sum = self.sum_counts(counts)
        counts_sum = [np.where(sum_exp > 1, sum_exp, np.ones(sum_exp.shape)) 
                      for sum_exp in counts_sum]   #to avoid division by small numbers
        
        count_mask = []
        t_mask = []
        counts_thresh = []
        counts_background = []
        
        if len(threshold) == 2:
            thresh_low = threshold[0]
            thresh_high = threshold[1]
        else:   #threshold can either be scalar or 2D vector, else will throw error
            thresh_low = threshold
            thresh_high = 2 * threshold
            
        for counts_exp, sum_exp, t in zip(counts, counts_sum, self.data.t_exp):
            ## intensity thresholding
            #create mask that has 1 if the summed counts of an exposure are above the threshold, and 0 otherwise
            mask_exp = np.where(np.logical_and(sum_exp > thresh_low, sum_exp < thresh_high), 
                                np.ones(sum_exp.shape, dtype=int), 
                                np.zeros(sum_exp.shape, dtype=int)) 
            
            ## filter out events with too high counts in central exposure
            mask_exp = np.where(np.logical_and(mask_exp, counts_exp[:,-1] < sum_exp*cfr_max), 
                                np.ones(sum_exp.shape, dtype=int), 
                                np.zeros(sum_exp.shape, dtype=int)) 
            
            ## variation-based thresholding
            if bin_var is not None:
                # for each exposure cycle, sum_var holds the local neighborhood of the count sum along the last axis
                sum_var = np.zeros((len(sum_exp) + bin_var, bin_var))  
                
                # initialize first and last bins with first and last datapoint; want size same as count traces, so need to fill up ends
                sum_var[:bin_var] = sum_exp[0]  
                sum_var[-bin_var:] = sum_exp[-1]
                for i in range(bin_var):    # looping over "bin axis" much more efficient than filling bins one by one (loop over count sum)
                    # each column is shifted by one; as result, bin is filled with entries [sum[k-bin_var/2], ..., sum[k], ..., sum[k+bin_var/2]]
                    sum_var[i:-(bin_var-i),i] = sum_exp
                
                # crop to proper length; need to distinguish between even/odd bin size
                if bin_var % 2 == 0:
                    sum_var = sum_var[bin_var//2:-(bin_var//2)]
                else:
                    sum_var = sum_var[bin_var//2:-(bin_var//2 + 1)]  
                     
                sum_noise = np.std(sum_var, axis=-1)/sum_exp  #calculate std. relative to sum for each data point
                
                # fig, ax = plt.subplots(1, 1, num=self.plot_ind, clear=True)
                # self.plot_ind += 1
                # ax.plot(sum_exp)
                # ax1 = ax.twinx()
                # ax1.plot(sum_noise, color='C1')
                
            else:
                sum_noise = np.zeros(sum_exp.shape)
            
            ## on-time thresholding
            switch_events = np.diff(mask_exp)     #equals 1 when switched on and -1 when switched off
            switch_on = np.nonzero(switch_events == 1)[0]     #indices of on-switching events
            switch_off = np.nonzero(switch_events == -1)[0]    #indices of off-switching events
            
            if mask_exp[0] == 1:   #case when  trace starts with emission event
                switch_on = np.hstack([0, switch_on])
            if mask_exp[-1] == 1:    #case when trace ends with active fluorophore
                switch_off = np.hstack([switch_off, counts_exp.shape[0] - 1])
            
            on_length = switch_off - switch_on
            off_length = np.hstack([switch_on, counts_exp.shape[0]]) - np.hstack([0, switch_off])
            # create array to encode length of on-event in mask
            length_mask = [np.zeros(off_length[0])]
            for on_event, off_event in zip(on_length, off_length[1:]):
                length_mask.append(on_event*np.ones(on_event))
                length_mask.append(np.zeros(off_event))
                        
            length_mask = np.hstack(length_mask)
            
            # modify mask to filter out "noisy" or too short on-events (flukes or diffusing dye in PAINT)
            mask_exp = np.where(np.logical_and(length_mask >= t_min/(self.parameters.n_tcp*self.parameters.t_tcp), sum_noise < thresh_var),
                                np.ones(sum_exp.shape, dtype=bool), 
                                np.zeros(sum_exp.shape, dtype=bool))

            mask_bg_exp = np.where(np.logical_and(length_mask == 0, sum_noise < thresh_var),
                                   np.ones(sum_exp.shape, dtype=bool), 
                                   np.zeros(sum_exp.shape, dtype=bool))
            # assume constant background for every experiment, but not for across TCP positions
            bg_exp = np.average(counts_exp[mask_bg_exp], axis=0)
            counts_background.append(bg_exp)
            print('background: ', bg_exp)
            if subtract_background:               
                counts_exp -= bg_exp
            
            # counts_thresh = [counts_exp[mask_exp] for counts_exp, counts_sum_exp in zip(counts, counts_sum)]
            counts_thresh.append(counts_exp[mask_exp])
            count_mask.append(mask_exp)
            t_mask.append(t[mask_exp])
            
        self.counts_thresh = counts_thresh
        self.count_mask = count_mask
        self.t_mask = t_mask
        self.counts_background = counts_background

        return counts_thresh, count_mask
    
    
    def bin_emission_events(self, counts=None, n_bin=1000):
        if counts is None:
            counts = self.counts_raw
            
        counts_sum = self.sum_counts(counts)
        
        thresh_mask = [np.array(mask_exp, dtype=int) for mask_exp in self.count_mask]
        events_binned = []
        sbr = []
        
        for i in range(len(counts)):
        # for counts_exp, sum_exp, mask_exp in zip(counts, counts_sum, thresh_mask):
            switch_events = np.diff(thresh_mask[i])     #equals 1 when switched on and -1 when switched off
            switch_on = np.nonzero(switch_events == 1)[0]     #indices of on-switching events
            switch_off = np.nonzero(switch_events == -1)[0]    #indices of off-switching events
            
            if thresh_mask[i][0] == 1:   #case when  trace starts with emission event
                switch_on = np.hstack([0, switch_on])
            if thresh_mask[i][-1] == 1:    #case when trace ends with active fluorophore
                switch_off = np.hstack([switch_off, counts[i].shape[0] - 1])
            
            events_exp = []
            t_cross = []
            sbr_exp = []
            for on_ind, off_ind in zip(switch_on, switch_off):
                counts_event = counts[i][on_ind+1:off_ind-1]   # +/-1 to exclude corner cases
                sum_event = counts_sum[i][on_ind+1:off_ind-1]
                cumsum_event = np.cumsum(sum_event)
                cross_ind = np.diff(cumsum_event//n_bin) == 1
                cross_ind = np.nonzero(cross_ind)[0] + 1    # +1 because diff decreases all indices by 1
                
                t_cross.append(self.data.t_exp[i][on_ind + cross_ind[:]])
                
                if len(cross_ind) > 0:
                    counts_binned = np.sum(counts_event[:cross_ind[0]], axis=0)
                    events_exp.append(counts_binned)
                    
                    bg_binned = self.counts_background[i]*cross_ind[0]
                    signal_binned = counts_binned - bg_binned
                    sbr_exp.append(signal_binned.sum()/bg_binned.sum())
                    
                for j in range(len(cross_ind) - 1):
                    counts_binned = np.sum(counts_event[cross_ind[j]:cross_ind[j+1]], 
                                           axis=0)
                    events_exp.append(counts_binned)
                    
                    bg_binned = self.counts_background[i]*(cross_ind[j+1] - cross_ind[j])
                    signal_binned = counts_binned - bg_binned
                    sbr_exp.append(signal_binned.sum()/bg_binned.sum())
                    # sbr_exp.append(signal_binned/bg_binned)
            
            events_binned.append(np.array(events_exp))
            sbr.append(np.array(sbr_exp))
            if len(t_cross) > 0:
                self.t_mask[i] = np.hstack(t_cross)
            else:
                self.t_mask[i] = np.empty(0)
        
        self.sbr = sbr
        return events_binned


    def estimate_position(self, counts_thresh=None, estimator='mle', estimator_param={}):
        
        localizations = []
        if counts_thresh is None:
            counts_thresh = self.counts_thresh
            
        for i in range(len(counts_thresh)):
            if counts_thresh[i].size > 0:    # empty count array would result in error
                if estimator == 'lms':
                    localizations.append(self.LMS(counts_thresh[i], ind_exp=i))
                elif estimator == 'mlms':
                    localizations.append(self.mLMS(counts_thresh[i], ind_exp=i, **estimator_param))
                elif estimator == 'mle':
                    localizations.append(self.MLE_numeric(counts_thresh[i], ind_exp=i, **estimator_param))
                else:
                    raise ValueError(f'"{estimator}" is not a valid estimator.')
                    # warnings.warn(f'"{estimator}" is not a valid estimator. Append empty array to localizations.')
                    # localizations.append([])
                
                if hasattr(self.data, 'offsets_tiles'):
                    localizations[-1] += self.data.offsets_tiles[i]
          
            else:
                localizations.append(np.empty((0, 2)))
            
        self.localizations_raw = localizations
        self.localizations = localizations    
        return localizations


    def _crop_arr(self, arr, x0, y0, size_x, size_y):
        x0 = int(round(x0))
        y0 = int(round(y0))
        size_x = int(round(size_x))
        size_y = int(round(size_y))
        return arr[x0:x0+size_x, y0:y0+size_y]
    
    
    def MLE_numeric(self, counts, plot_p=True, ind_exp=0):
        '''numeric MLE for arbitrary pattern (beam_shape() must have shape_param and center as 1st and 2nd arguments)'''
        if not hasattr(self.data, 'target_coordinates'):
            self.data.create_target_coordinates()
            
        r_list = []
        for i in range(len(counts)):
            # calculate self.p_grid in case it doesn't (yet) exist
            if not hasattr(self, 'p_log_grid'):
                if self.psf_data is not None:
                    crop_size = np.array(self.psf_data.psf.shape) - self.parameters.L/self.psf_data.px_size
                    # beam_target_coord = np.array([self._crop_arr(self.psf, 1 + (self.parameters.L/2 - tc[1])/self.px_size_grid, 1 + (self.parameters.L/2 + tc[1])/self.px_size_grid, 
                    #                                         1 + (self.parameters.L/2 - tc[0])/self.px_size_grid, 1 + (self.parameters.L/2 + tc[0])/self.px_size_grid)
                    #                              for tc in self.target_coordinates])
                    target_coord_grid = np.array([self._crop_arr(self.psf_data.psf, 
                                                                (self.parameters.L/2 - tc[1])/self.psf_data.px_size, 
                                                                (self.parameters.L/2 - tc[0])/self.psf_data.px_size,
                                                                crop_size[1], crop_size[0])
                                                  for tc in self.data.target_coordinates])
                
                else:
                    raise AttributeError('No psf set.')
                    
                p_grid = target_coord_grid/sum(target_coord_grid)    #normalize to obtain p-parameters
                p_grid += 1e-12     #to avoid having 0 in center which may lead to numerical instability
                self.p_grid = p_grid
                self.p_log_grid = np.log(p_grid)
                
                if plot_p:
                    self.plot_ind = 10
                    fig, ax = plt.subplots(1, 5, num=self.plot_ind, clear=True)
                    self.plot_ind += 1
                    for j in range(len(p_grid)):
                        ax[j].imshow(self.p_log_grid[j])
                
            # # log_l = sum([counts[i]*np.log(p[i]) for i in range(len(counts))])
            # log_l = np.tensordot(counts[i], self.p_log_grid, axes=(0,0))     #compute log-likelihood function sum(n_i*ln(p_i))
            
            # # l_argmax = np.nonzero(log_l == log_l.max(axis=(1, 2))[:, np.newaxis, np.newaxis])
            # l_argmax = np.nonzero(log_l == log_l.max())
            # # r_estimate = np.array([l_argmax[2], l_argmax[1]]).transpose()
            # r_list.append([l_argmax[1][0], l_argmax[0][0]])
            
            # # r_estimate = np.array(np.unravel_index(log_l.argmax(), log_l.shape))[::-1]    #np.unravel_index converts flat index (returned by argmax) to coordinate index
            
            
            
            p_corr = self._p_correct_background_mle(self.p_grid, self.sbr[ind_exp][i])
            # p_corr = self.p_grid
            # for j in range(len(p_grid)):
            #     ax[j].imshow(p_corr[j])
            
            # counts[i] = np.array([0.8*len(counts), 0.6*len(counts), 0.4*len(counts), i])
            log_l = sum([counts[i,k]*np.log(p_corr[k]) for k in range(counts.shape[1])])
            l_argmax = np.nonzero(log_l == np.nanmax(log_l))
            r_list.append([l_argmax[1][0], l_argmax[0][0]])
            
            if plot_p and (i == 100): 
                ax[-1].imshow(log_l)
 
        r_arr = np.asarray(r_list)
        r_arr = r_arr * self.psf_data.px_size - self.psf_data.grid_size/2 + self.parameters.L/2
        
        return r_arr 
      
   
    def LMS(self, counts, ind_exp=0):
        '''
        least mean square estimator according to Balzarotti et al. (SI 3.2.1)        

        Parameters
        ----------
        counts : TYPE
            DESCRIPTION.

        Returns
        -------
        position estimate

        ''' 
        p = counts/np.sum(counts, axis=1)[:,np.newaxis]     #newaxis required for proper broadcasting
        r_estimate = self.parameters.L/2/(1 - self.parameters.L**2*np.log10(2)/self.parameters.fwhm**2)* \
                        np.vstack([-p[:,2] + (p[:,0] + p[:,1])/2, np.sqrt(3)/2*(p[:,1] - p[:,0])])
        
        return r_estimate.transpose()
    
    
    def mLMS(self, counts, ind_exp=0, beta=None):
        '''order k deduced from length of beta, beta default parameters taken from Balzarotti et al. (not found in Gwosch et al.)'''
        if beta is None: 
            # beta = [1.27, 3.8]    #values from Balzarotti et al.
            beta = [0.4, 4, 6]
        if not hasattr(self.data, 'target_coordinates'):
            self.data.create_target_coordinates()
    
        k = len(beta)
        p = counts/np.sum(counts, axis=1)[:,np.newaxis]     #newaxis required for proper broadcasting
        r_estimate = -1/(1 - self.parameters.L**2*np.log10(2)/self.parameters.fwhm**2)* \
                        sum([beta[j]*p[:,-1]**j for j in range(k)])[:,np.newaxis]* \
                        (p[:,:self.parameters.n_tcp-1] @ self.data.target_coordinates[:self.parameters.n_tcp-1])
        return r_estimate

      
    def _p_correct_background_mle(self, p, sbr):
        # formula (S30) from Balzarotti et al. (2016)
        # compatible with LMS: p is 2D array, sbr is 1D array; and MLE: p is 3D array, sbr is scalar 
        p = sbr/(sbr + 1) * p + 1/(1 + sbr) * 1/self.parameters.n_tcp
        return p
    
   
    def crop_localizations(self, x_min, x_max, y_min, y_max, localizations=None):
        if localizations is None:
            localizations = self.localizations
        loc_crop = []
        for i in range(len(localizations)):
            crop_ind = (localizations[i][:,0] > x_min) * (localizations[i][:,0] < x_max) * \
                       (localizations[i][:,1] > y_min) * (localizations[i][:,1] < y_max) 
            loc_crop.append(localizations[i][crop_ind])
            self.t_mask[i] = self.t_mask[i][crop_ind]
            
        self.localizations = loc_crop
        return loc_crop
        
    
    def subtract_drifts_fit(self, localizations=None, k_fit=5, fit_separate=True):
        '''
        k_fit: degree of polynomials to fit to data
        fit_separate: if True, drift functions are applied to experiments seperately; else avg. to all 
        '''
        if localizations is None:
            localizations = self.localizations
    
        loc_corrected = []
        p_fit = []
        for i in range(len(localizations)):
            if localizations[i].size > 0:
                t_fit = self.t_mask[i]
                t_fit = (t_fit - min(t_fit))/(max(t_fit) - min(t_fit))            # when values grow too large fitting doesn't work, so don't take t_mask directly
                p_fit_exp = np.polyfit(t_fit, localizations[i], k_fit)
                if fit_separate:
                    loc_fit_x = np.sum([p_fit_exp[j,0]*t_fit**(k_fit - j) for j in range(k_fit+1)], axis=0)
                    loc_fit_y = np.sum([p_fit_exp[j,1]*t_fit**(k_fit - j) for j in range(k_fit+1)], axis=0)
                    loc_fit_exp = np.vstack([loc_fit_x, loc_fit_y]).transpose()
                    loc_corrected.append(localizations[i] - loc_fit_exp + p_fit_exp[-1])    # ad p_fit[-1] because do not want localizations to be centered around 0
                p_fit.append(p_fit_exp)
            else:
                loc_corrected.append(np.empty((0, 2)))    # shape required to concatenate with  other exp
                p_fit.append(np.empty((0, 2)))
    
        if not fit_separate:
            p_fit = np.average(p_fit, axis=0)
            for i in range(len(localizations)):
                if localizations[i].size > 0:
                    t_fit = self.t_mask[i]
                    t_fit = (t_fit - min(t_fit))/(max(t_fit) - min(t_fit))            # when values grow too large fitting doesn't work, so don't take t_mask directly
                    loc_fit_x = np.sum([p_fit[j,0]*t_fit**(k_fit - j) for j in range(k_fit+1)], axis=0)
                    loc_fit_y = np.sum([p_fit[j,1]*t_fit**(k_fit - j) for j in range(k_fit+1)], axis=0)
                    loc_fit_exp = np.vstack([loc_fit_x, loc_fit_y]).transpose()
                    loc_corrected.append(localizations[i] - loc_fit_exp + p_fit[-1])    # ad p_fit[-1] because averaging might not make sense for constant term of polyfit
                else:
                    loc_corrected.append(np.empty((0, 2)))
            p_fit = [p_fit]
            
        self.localizations = loc_corrected
        self.p_drift = p_fit
        return loc_corrected, p_fit
    
    
    def subtract_drifts_fiducials(self, localizations=None, k_on=0.8, k_exp=None, d_pos=10, k_fit=5):
        '''
        k_on: minimum fraction in ON state to be detected as fidcucials
        k_exp: manually select experiment indices with fiducials (ignored if None)
        d_pos: distance at which subsequent localizations are registered as jumps rather than noise
        k_fit: degree of polynomial to fit to data
        '''
        if localizations is None:
            localizations = self.localizations
    
        if k_exp:
            ind_exp = k_exp
        else:
            ind_exp = range(len(localizations))
        p_fit = []
        for i in ind_exp:
            # fiducial might be at exp i if emission is ON most of the time
            if np.count_nonzero(self.count_mask[i])/self.count_mask[i].size > k_on:
                d_loc = np.gradient(localizations[i], axis=0)
                d_loc = np.linalg.norm(d_loc, axis=1)
                # exclude exp i if there are too many large jumps between localizations (allow a few to account for outliers)
                if np.count_nonzero(d_loc > d_pos)/d_loc.size < 0.01:
                    loc_med = np.median(localizations[i], axis=0)
                    loc_centered = localizations[i] - loc_med
                    d_med = np.sqrt(np.sum(loc_centered**2, axis=1))   #d_med is Euklidean distance from median
                    mdev = np.median(d_med)
                    s_med = d_med/mdev if mdev else 0.      # standardized deviation from median (robust to outliers)
                    
                    k_outlier = 5
                    t_fit = self.t_mask[i][s_med < k_outlier]
                    t_fit = (t_fit - min(t_fit))/(max(t_fit) - min(t_fit))            # when values grow too large fitting doesn't work, so don't take t_mask directly
                    loc_fit = localizations[i][s_med < k_outlier]
                    if loc_fit.size > 0:
                        p_fit.append(np.polyfit(t_fit, loc_fit, k_fit))
                    else:
                        p_fit.append(np.empty(k_fit))
                        warnings.warn(f'exp {i} did not pass outlier test')
                        
        if len(p_fit) == 0:
            warnings.warn('No fiducials found. Could not apply drift correction')
            loc_corr = localizations
            p_fit_avg = p_fit
        else:
            p_fit_avg = np.average(p_fit, axis=0)
            
            loc_corr = []
            for i in range(len(localizations)):
                if localizations[i].size > 0:
                    t_fit = self.t_mask[i]
                    t_fit = (t_fit - min(t_fit))/(max(t_fit) - min(t_fit))            # when values grow too large fitting doesn't work, so don't take t_mask directly
                    loc_fit_x = np.sum([p_fit_avg[j,0]*t_fit**(k_fit - j) for j in range(k_fit+1)], axis=0)
                    loc_fit_y = np.sum([p_fit_avg[j,1]*t_fit**(k_fit - j) for j in range(k_fit+1)], axis=0)
                    loc_fit_exp = np.vstack([loc_fit_x, loc_fit_y]).transpose()
                    loc_corr.append(localizations[i] - loc_fit_exp + p_fit_avg[-1])    # ad p_fit[-1] because averaging might not make sense for constant term of polyfit
                else:
                    loc_corr.append(np.empty(0, 2))
                    
        self.localizations = loc_corr
        p_fit_avg = [p_fit_avg]
        return loc_corr, p_fit_avg
    
    
    
    



