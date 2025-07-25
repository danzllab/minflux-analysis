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
    '''Initialize minflux localization with parameters and data. Optional: specify psf (required for MLE localization).'''
    def __init__(self, parameters, data, psf_data=None):
        # ensure same data format if no grid was generated; list with entries of shape (n_cycles, n_tcp) per tip/tilt exposure; each entry in list called "_exp" for experiment
        if hasattr(data, 'counts_tiles'):
            self.counts_raw = data.counts_tiles
            self.counts_processed = data.counts_tiles
        else:
            self.counts_raw = [data.counts_raw]
            self.counts_processed = [data.counts_raw]
        if not hasattr(data, 't_exp'):  # create default timestamps if not previously specified.
            warnings.warn('No timestamps found. Create with dt = 1 s.')
            data.create_timestamps(1)
        
        self.parameters = parameters
        self.data = data
        self.psf_data = psf_data
        
    
    def remove_activation_counts(self, counts=None):
        '''Remove cycles when conditional activation was active from count traces; either specify counts or use processed counts.'''
        if counts is None:
            counts = self.counts_processed
        
        counts_filtered = []
        for counts_exp in counts:
            act_cycle = np.count_nonzero(counts_exp, axis=1) <= 1    #convention of dcs program: during activation first exposure = sum of counts, others 0
            counts_filtered_exp = np.where(np.vstack((act_cycle, )*self.parameters.n_tcp).transpose(), 
                                           np.transpose((counts_exp[:,0]/self.parameters.n_tcp, )*self.parameters.n_tcp),   # replace activation cycles by avg. counts across whole TCP for each iteration
                                           counts_exp)  #if no activation cycle, retain counts
            counts_filtered.append(counts_filtered_exp)
        
        self.counts_processed = counts_filtered
        return counts_filtered
 
    
    def bin_counts(self, counts=None, k_bin=5):
        '''Bin counts  of multiple cycles to improve precision; sum k_bin data points together; either specify counts or use processed counts.'''
        if counts is None:
            counts = self.counts_processed 
        
        counts_binned = []
        t_exp = [t[::k_bin] for t in self.data.t_exp]   # adjust time stamps to match binned counts
        for i, counts_exp in enumerate(counts):
            stub = counts_exp.shape[0]%k_bin    # stub: extra elements if number of cycles not divisible by k_bin
            if stub: # if stub exists, remove
                t_exp[i] = t_exp[i][:-1]
                counts_exp = counts_exp[:-stub]     #crop last entries to have number of cycles fit k_bin
            counts_exp_binned = counts_exp.reshape(-1, k_bin, self.parameters.n_tcp).sum(axis=1)    # bin counts as specified
            counts_binned.append(counts_exp_binned)
            
        self.counts_processed = counts_binned   # binned counts new processed counts
        
        self.parameters.t_tcp *= k_bin      # adjust effective exposure time
        self.data.t_exp = t_exp
        
        return counts_binned

        
    def filter_counts(self, filter_type, t_filter, counts=None, normalize=True):
        '''
        Filter counts traces to remove noise; either specify counts or use processed counts.

        Parameters
        ----------
        filter_type : currently "box", "gauss", "triangle implemented"
        t_filter : time constant of filter; width of box, sigma of gauss, or rise-/fall-time for triangle
        counts : specify count traces or use processed counts
        normalize : normalize via division of filter weights by their sum

        Returns
        -------
        counts_filtered
        '''
        if counts is None:
            counts = self.counts_processed
        
        if filter_type == 'gauss': 
            filter_weights = np.exp(-np.arange(-250, 250)**2/(2*t_filter**2))
        elif filter_type == 'box': 
            filter_weights = np.ones(t_filter)
        elif filter_type == 'triangle': 
            filter_weights = np.hstack((np.arange(t_filter), np.arange(t_filter, 0, -1)))   #stack ascending and decending slopes
        else:
            raise ValueError("'filter_type' must be 'gauss', 'box' or 'triangle'.")
        
        if normalize: filter_weights = filter_weights/sum(filter_weights)
        filter_weights = filter_weights.reshape(-1, 1)  #reshape f_filter to convolve along axis 0
        counts_filtered = [fftconvolve(counts_exp, filter_weights, mode='same') for counts_exp in counts]   # mode "same" gives edge effects but no phase (i.e., output same length as input); accept for now and reevaluate if needed
        
        self.counts_processed = counts_filtered
        return counts_filtered
    
    
    def sum_counts(self, counts=None):
        '''Sum counts over all TCP-exposures for each cycle; specify counts or use processed counts.'''
        if counts is None:
            counts = self.counts_processed          
        counts_sum = [np.sum(counts_exp, axis=1) for counts_exp in counts]
        return counts_sum
    
    
    def threshold_counts(self, threshold, counts=None, cfr_max=0.25, t_min=1, 
                         bin_var=None, thresh_var=0.1, subtract_background=False):
        '''
        Threshold counts to extract single-molecule emission events.

        Parameters
        ----------
        threshold : count threshold; either list containing lower and upper or single value (lower threshold; multiplied by 2 to obtain upper)
        counts : specify count traces or use processed counts
        cfr_max : maximum center frequency ratio (fraction of counts in center exposure)
        t_min : minimum on-time in s
        bin_var : how many cycles to bin for calculating local variance of count levels; if none, no variance-based thresholding is performed
        thresh_var : threshold for local variance of summed counts relative to count levels
        subtract_background : boolean indicating if background is subtracted from count traces (not modeled!); not recommended

        Returns
        -------
        counts_thresh : counts corresponding to time points when single-molecule emission was detected
        count_mask : mask indicating time points when single-molecule emission was detected

        '''
        
        if counts is None:
            counts = self.counts_processed        
            
        counts_sum = self.sum_counts(counts)    # sum across TCP-exposures
        counts_sum = [np.where(sum_exp > 1, sum_exp, np.ones(sum_exp.shape)) 
                      for sum_exp in counts_sum]   #to avoid division by small numbers
        
        count_mask = []
        t_mask = []
        counts_thresh = []
        counts_background = []
        
        if hasattr(threshold, '__iter__'):
            thresh_low = threshold[0]
            thresh_high = threshold[1]
        else:   # if single value is given, assum upper threshold to be twice of (given) lower threshold
            thresh_low = threshold
            thresh_high = 2 * threshold
            
        for counts_exp, sum_exp, t in zip(counts, counts_sum, self.data.t_exp):
            ## intensity thresholding
            #create mask that has 1 if the summed counts of an exposure are within the thresholds, and 0 otherwise
            mask_exp = np.where(np.logical_and(sum_exp > thresh_low, sum_exp < thresh_high), 
                                np.ones(sum_exp.shape, dtype=int), 
                                np.zeros(sum_exp.shape, dtype=int)) 
            
            ## filter out events with too high counts in central exposure
            mask_exp = np.where(np.logical_and(mask_exp, counts_exp[:,-1] < sum_exp*cfr_max),   # central exposure in last count column!
                                np.ones(sum_exp.shape, dtype=int), 
                                np.zeros(sum_exp.shape, dtype=int)) 
            
            ## variation-based thresholding
            if bin_var is not None:
                # for each TCP-cycle, sum_var holds the summed counts corresponding to surrounding time point s
                sum_var = np.zeros((len(sum_exp) + bin_var, bin_var))       # time points along axis 0, filled up with summed counts in temporal proximity along axis 1
                # initialize first and last bins with first and last datapoint, respectively; want size same as count traces, so need to fill up ends
                sum_var[:bin_var] = sum_exp[0]  
                sum_var[-bin_var:] = sum_exp[-1]
                for i in range(bin_var):    # looping over "bin axis" much more computationally efficient than filling bins one by one (i.e., looping over count sum)
                    # each column is shifted by one; as result, bin is filled with entries [sum[k-bin_var/2], ..., sum[k], ..., sum[k+bin_var/2]]
                    sum_var[i:-(bin_var-i), i] = sum_exp
                
                # crop to proper length; distinguish between even/odd bin size
                if bin_var % 2 == 0:
                    sum_var = sum_var[bin_var//2:-(bin_var//2)]
                else:
                    sum_var = sum_var[bin_var//2:-(bin_var//2 + 1)]  
                     
                sum_noise = np.std(sum_var, axis=-1)/sum_exp  # calculate standard deviation relative to sum for each data point - to be used for thresholding
                
            else:   # if bin_var not given, set variance to 0 to bypass variance-based thresholding
                sum_noise = np.zeros(sum_exp.shape)
            
            ## on-time thresholding
            switch_events = np.diff(mask_exp)     # extract time points when switching occurs (from count levels and cfr thresholding); 1 when on-switching and -1 when off-switching
            switch_on = np.nonzero(switch_events == 1)[0]     # indices of on-switching events
            switch_off = np.nonzero(switch_events == -1)[0]    # indices of off-switching events
            # format mask_exp to ensure that it starts and ends in the non-emissive state
            if mask_exp[0] == 1:   # when  trace starts with emission event, assume that it switches on at t = 0 s
                switch_on = np.hstack([0, switch_on])
            if mask_exp[-1] == 1:    # when trace ends with active fluorophore, assume that it switches off when experiments ends
                switch_off = np.hstack([switch_off, counts_exp.shape[0] - 1])
            
            # extract lengths of instances with active/inactive emitter
            on_length = switch_off - switch_on      # arrays formatted previously to ensure on-switching always deteced before off-switching
            off_length = np.hstack([switch_on, counts_exp.shape[0]]) - np.hstack([0, switch_off])   # stack last/first datapoints in end/beginning to reverse order
            # create array to encode length of on-event in mask: for every time point, 0 if no emission event detected, on-length of corresponding event (in TCP-cyles) if emission detected 
            length_mask = [np.zeros(off_length[0])]     #initialize with first off-event
            for on_event, off_event in zip(on_length, off_length[1:]):  # build length mask by alternating on- and off-events 
                length_mask.append(on_event*np.ones(on_event))      # append value of "on"-cycles once for every cycle
                length_mask.append(np.zeros(off_event))             # append 0s while no emission detected
            length_mask = np.hstack(length_mask)    # stack to flatten mask
            
            # modify previously obtained mask to filter out too short (flukes or diffusing dye in PAINT) or "noisy" on-events
            mask_exp = np.where(np.logical_and(length_mask >= t_min/(self.parameters.n_tcp*self.parameters.t_tcp), sum_noise < thresh_var),
                                np.ones(sum_exp.shape, dtype=bool), 
                                np.zeros(sum_exp.shape, dtype=bool))
            
            # define background as time point when summed counts and their local variance are below respective threshold
            mask_bg_exp = np.where(np.logical_and(sum_exp < thresh_low, sum_noise < thresh_var),
                                   np.ones(sum_exp.shape, dtype=bool), 
                                   np.zeros(sum_exp.shape, dtype=bool))
            # assume constant background for every experiment, but not for across TCP positions
            if np.count_nonzero(mask_bg_exp):
                bg_exp = np.average(counts_exp[mask_bg_exp], axis=0)
                counts_background.append(bg_exp)
            else:
               warnings.warn('No background counts detected. Set to 1e-32.')
               bg_exp = 1e-32*np.ones(counts_exp.shape[1])  #set to infinitesimally small value above 0
               counts_background.append(bg_exp)
                
            if subtract_background:               
                counts_exp -= bg_exp
                    
            # append thresholded counts, threshold mask, and time points at which active emitter was detected
            counts_thresh.append(counts_exp[mask_exp])
            count_mask.append(mask_exp)
            t_mask.append(t[mask_exp])
            
        self.counts_thresh = counts_thresh
        self.count_mask = count_mask
        self.t_mask = t_mask
        self.counts_background = counts_background

        return counts_thresh, count_mask
    
    
    def bin_emission_events(self, counts=None, n_bin=1000):
        '''
        Bin single-molecule emission events according to specified photon number.

        Parameters
        ----------
        counts : specify count traces or use raw counts
        n_bin : how many counts to bin

        Returns
        -------
        events_binned : Binned single-molecule counts.

        '''
        if counts is None:
            counts = self.counts_raw
            
        counts_sum = self.sum_counts(counts)
        
        thresh_mask = [np.array(mask_exp, dtype=int) for mask_exp in self.count_mask]
        events_binned = []
        sbr = []
        
        for i in range(len(counts)):
            # extract on- and off-switching time points analogous to def threshold_counts()
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
                counts_event = counts[i][on_ind+1:off_ind-1]   # +/-1 to exclude cycles during which on-/off-switching happened, potentially distorting count ratios
                sum_event = counts_sum[i][on_ind+1:off_ind-1]
                cumsum_event = np.cumsum(sum_event)     # calculate cumulative sum for given emission event
                cross_ind = np.diff(cumsum_event//n_bin) == 1   # determine indices where accumulated counts cross a multiple of the bin number
                cross_ind = np.nonzero(cross_ind)[0] + 1    # exract indices; +1 because applying np.diff decreases all indices by 1
                
                t_cross.append(self.data.t_exp[i][on_ind + cross_ind[:]])
                
                if len(cross_ind) > 0:
                    counts_binned = np.sum(counts_event[:cross_ind[0]], axis=0)     # bin first ~n_bin counts of emission event
                    events_exp.append(counts_binned)
                    
                    # scale background to account for numer of cycles summed up for binning and calculate signal-to-background ratio
                    bg_binned = self.counts_background[i]*cross_ind[0]
                    signal_binned = counts_binned - bg_binned
                    sbr_exp.append(signal_binned.sum()/bg_binned.sum())
                    
                for j in range(len(cross_ind) - 1):     # if multiple bins exist for emission event, iterate over them
                    counts_binned = np.sum(counts_event[cross_ind[j]:cross_ind[j+1]], 
                                           axis=0)
                    events_exp.append(counts_binned)
                    
                    bg_binned = self.counts_background[i]*(cross_ind[j+1] - cross_ind[j])
                    signal_binned = counts_binned - bg_binned
                    sbr_exp.append(signal_binned.sum()/bg_binned.sum())
                    # sbr_exp.append(signal_binned/bg_binned)
            
            events_binned.append(np.array(events_exp))
            sbr.append(np.array(sbr_exp))
            if len(t_cross) > 0:    # modify mask of time points
                self.t_mask[i] = np.hstack(t_cross)
            else:
                self.t_mask[i] = np.empty(0)
        
        self.sbr = sbr
        return events_binned


    def estimate_position(self, counts_thresh=None, estimator='mle', estimator_param={}):
        '''
        Perform position estimation using MLE or (m)LMS. 

        Parameters
        ----------
        counts_thresh : specify or use thresholded counts.
        estimator : lms, mlms or mle.
        estimator_param : dictionary of estimator-specific parameters.

        Returns
        -------
        localizations
        '''
        localizations = []
        if counts_thresh is None:
            counts_thresh = self.counts_thresh
            
        for i in range(len(counts_thresh)):     # iterate over experiments/tip-tilt positions; for every iteration, estimate coordinates for all sets of counts with one function call
            if counts_thresh[i].size > 0:    # empty count array would result in error
                if estimator == 'lms':
                    localizations.append(self.LMS(counts_thresh[i], ind_exp=i))
                elif estimator == 'mlms':
                    localizations.append(self.mLMS(counts_thresh[i], ind_exp=i, **estimator_param))
                elif estimator == 'mle':
                    localizations.append(self.MLE_numeric(counts_thresh[i], ind_exp=i, **estimator_param))
                else:
                    raise ValueError(f'"{estimator}" is not a valid estimator.')
                    
                if hasattr(self.data, 'offsets_tiles'):     # add offsets produced during grid generation
                    localizations[-1] += self.data.offsets_tiles[i]
          
            else:   # if counts in experiment, append empty array
                localizations.append(np.empty((0, 2)))
            
        self.localizations_raw = localizations
        self.localizations = localizations    
        return localizations


    def MLE_numeric(self, counts, plot_mle=False, ind_exp=0):
        '''
        Numeric maximum likelihood estimation for 

        Parameters
        ----------
        counts : photon counts; sets of counts along axis 0 with TCP-exposures along axis 1
        plot_mle : boolean indicating whether to plot p-grids and localization for every 1000th localization
        ind_exp : index of experiment; to use corresponding SBRs

        Returns
        -------
        r_arr : Array of estimated coordinates (in nm); localizations along axis 0 with x- and y-coordinates along axis 1
        '''
        if not hasattr(self.data, 'target_coordinates'):
            self.data.create_target_coordinates()
           
        # calculate self.p_grid in case it doesn't (yet) exist; same for all localizations, so only need to generate once
        if not hasattr(self, 'p_grid'):
            if self.psf_data is not None:
                crop_size = np.array(self.psf_data.psf.shape) - self.parameters.L/self.psf_data.px_size
                # cropping effectively shifts beam within resulting grid
                target_coord_grid = np.array([self._crop_arr(self.psf_data.psf, 
                                                            (self.parameters.L/2 - tc[1])/self.psf_data.px_size, 
                                                            (self.parameters.L/2 - tc[0])/self.psf_data.px_size,
                                                            crop_size[1], crop_size[0])
                                              for tc in self.data.target_coordinates])
            
            else:
                raise AttributeError('No psf set.')
                
            p_grid = target_coord_grid/sum(target_coord_grid)    # divide intensities by sum of TCP-exposures for each pixel to obtain p-parameters
            p_grid += 1e-32     #to avoid having 0 in center which may lead to numerical instability
            self.p_grid = p_grid
                     
        r_list = []
        for i in range(len(counts)):
            if hasattr(self, 'sbr'):    # correct p-parameters for background (if given)
                p_corr = self._p_correct_background(self.p_grid, self.sbr[ind_exp][i])
            else:
                warnings.warn('No signal-to-background ratios set. Localizations may be biased.')
                p_corr = self.p_grid
            p_corr_log = np.log(p_corr)
            
            # perform position estimation
            log_l = sum([counts[i,k]*p_corr_log[k] for k in range(counts.shape[1])])    # calculate log-likelihood 
            l_argmax = np.nonzero(log_l == np.nanmax(log_l))
            r_list.append([l_argmax[1][0], l_argmax[0][0]])     # append position of maximum log-likelihood to localization list
            
            if plot_mle and (i%1000 == 0):  # if specified, plot p-grid, log-likelihood and localization for every 1000th iteration
                self.plot_ind = 10
                fig, ax = plt.subplots(1, len(p_corr_log)+1, num=self.plot_ind, clear=True)
                for j in range(len(p_corr_log)):    #plot corrected p_log grids
                    ax[j].imshow(p_corr_log[j], cmap='cividis')#, vmin=np.min(p_corr_log[j]), vmax=np.max(p_corr_log[j]))
                    
                ax[-1].imshow(log_l, cmap='cividis')    #plot grid of log-likelihoods
                ax[-1].scatter(l_argmax[1][0], l_argmax[0][0], color='k')   #plot localization
                for a in ax:    # axes in units of pixels; would have to specify or convert to make clear; omitting is "lazy" solution (since plotting is intended mainly for internal inspection)
                    a.set_xticks([])
                    a.set_yticks([])
                plt.tight_layout()
                counts_str = str(counts[i]).split()
                fig.savefig(self.parameters.file_name + f'_log-likelihood_iteration{i}_counts{"-".join(counts_str)[1:-1]}.pdf', transparent=True)
 
        r_arr = np.asarray(r_list)
        r_arr = r_arr * self.psf_data.px_size - self.psf_data.grid_size/2 + self.parameters.L/2     # convert to nm and center around 0
        
        return r_arr 
      
   
    def LMS(self, counts, ind_exp=0):
        '''
        linearized least mean square estimator according to Balzarotti et al., Science (2017) (SI 3.2.1)        

        Parameters
        ----------
        counts : photon counts; sets of counts along axis 0 with TCP-exposures along axis 1
        ind_exp : index of experiment; to use corresponding SBRs

        Returns
        -------
        r_estimate : Array of estimated coordinates (in nm); localizations along axis 0 with x- and y-coordinates along axis 1

        ''' 
        if not hasattr(self.data, 'target_coordinates'):
            self.data.create_target_coordinates()
            
        p = counts/np.sum(counts, axis=1)[:,np.newaxis]     #newaxis required for proper broadcasting of sum
        if hasattr(self, 'sbr'):
            p = self._p_correct_background(p, self.sbr[ind_exp][:,np.newaxis])
        else:
            warnings.warn('No signal-to-background ratios set. Localizations may be biased.')
        
        # equivalent expressions; Equations S49, S50 in Balzarotti et al., Science (2017)
        # r_estimate = self.parameters.L/2 * 1/(1 - np.log10(2)*(self.parameters.L/self.parameters.fwhm)**2)* \
        #                 np.vstack([-p[:,2] + (p[:,0] + p[:,1])/2, np.sqrt(3)/2*(p[:,1] - p[:,0])])
        # r_estimate = r_estimate.transpose()
             
        r_estimate = -1/(1 - self.parameters.L**2*np.log10(2)/self.parameters.fwhm**2)* \
                    (p[:,:self.parameters.n_tcp-1] @ self.data.target_coordinates[:self.parameters.n_tcp-1])
        
        return r_estimate
    
    
    def mLMS(self, counts, ind_exp=0, beta=None):
        '''
        modified least mean square estimator according to Balzarotti et al., Science (2017) (SI 3.2.2)        

        Parameters
        ----------
        counts : photon counts; sets of counts along axis 0 with TCP-exposures along axis 1
        ind_exp : index of experiment; to use corresponding SBRs
        beta : correction factors; (beta_0, beta_1, ..., beta_k) --> sum(beta_k * p^k for k in range(len(beta))); k deduced from length of beta; if none, use parameters from Balzarotti et al., Science (2017)

        Returns
        -------
        r_estimate : Array of estimated coordinates (in nm); localizations along axis 0 with x- and y-coordinates along axis 1
        '''
        if beta is None: 
            beta = [1.27, 3.8]    #values from Balzarotti et al.
            # beta = [0.4, 4, 6]
        if not hasattr(self.data, 'target_coordinates'):
            self.data.create_target_coordinates()
    
        p = counts/np.sum(counts, axis=1)[:,np.newaxis]     #newaxis required for proper broadcasting of sum
        if hasattr(self, 'sbr'):
            p = self._p_correct_background(p, self.sbr[ind_exp][:,np.newaxis])
        else:
            warnings.warn('No signal-to-background ratios set. Localizations may be biased.')
            
        k = len(beta)
        # Equation S51 in Balzarotti et al., Science (2017)
        r_estimate = -1/(1 - self.parameters.L**2*np.log10(2)/self.parameters.fwhm**2)* \
                        sum([beta[j]*p[:,-1]**j for j in range(k)])[:,np.newaxis]* \
                        (p[:,:self.parameters.n_tcp-1] @ self.data.target_coordinates[:self.parameters.n_tcp-1])
        
        return r_estimate


    def crop_localizations(self, x_min, x_max, y_min, y_max, localizations=None):
        '''Crop localizations (specified or use saved) to ROI (x_min, x_max, y_min, y_max).'''
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
        Post hoc drift correction by fitting polynomial function to localizations and subtracting.

        Parameters
        ----------
        localizations : specify or use saved localizations
        k_fit : order of polynomial to fit
        fit_separate : fit experiments separately or use the same polynomial for all.

        Returns
        -------
        loc_corrected : Drift-corrected localizations.
        p_fit : Obtained polynomial parameters (p_k, p_{k-1}, ..., p_1, p_0)
        '''
        if localizations is None:
            localizations = self.localizations
    
        loc_corrected = []
        p_fit = []
        for i in range(len(localizations)):
            if localizations[i].size > 0:   # only fit if there are localizations in grid position/experiment
                t_fit = self.t_mask[i]
                t_fit = (t_fit - min(t_fit))/(max(t_fit) - min(t_fit))      # when values grow too large fitting doesn't converge, so scale to (0, 1)
                p_fit_exp = np.polyfit(t_fit, localizations[i], k_fit)      # obtain polynomial fit parameters (separate for localization axes)
                if fit_separate:    # if specified, correct drifts with separate polynomials
                    loc_fit_x = np.sum([p_fit_exp[j,0]*t_fit**(k_fit - j) for j in range(k_fit+1)], axis=0)     # generate polynomial for x
                    loc_fit_y = np.sum([p_fit_exp[j,1]*t_fit**(k_fit - j) for j in range(k_fit+1)], axis=0)     # generate polynomial for y
                    loc_fit_exp = np.vstack([loc_fit_x, loc_fit_y]).transpose()
                    loc_corrected.append(localizations[i] - loc_fit_exp + p_fit_exp[-1])    # subtract polynomial from localizations; add p_fit[-1] because do not want localizations to be centered around 0
                p_fit.append(p_fit_exp)
            else:
                loc_corrected.append(np.empty((0, 2)))    # shape required to concatenate with  other experiments
                p_fit.append(np.empty((0, 2)))
    
        if not fit_separate:    # if specified, correct drifts with common polynomial
            p_fit = np.average(p_fit, axis=0)   # not most elegant way to average; might modify if performance turns out to be impaired
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
        Post hoc drift correction by detecting fiduials and using them for drift correction.

        Parameters
        ----------
        k_on : minimum fraction in ON state to be detected as fiducials
        k_exp : manually select experiment indices with fiducials; if none, use all
        d_pos : distance at which subsequent localizations are registered as jumps rather than noise (in nm)
        k_fit : degree of polynomial to fit to data

        Returns
        -------
        loc_corr : Drift-corrected localizations.
        p_fit_avg : Obtained polynomial parameters (p_k, p_{k-1}, ..., p_1, p_0)
        '''
        if localizations is None:
            localizations = self.localizations
    
        if k_exp:   # set experiment indices of interest
            ind_exp = k_exp
        else:
            ind_exp = range(len(localizations))
        
        p_fit = []
        for i in ind_exp:
            # fiducial might be at exp i if emission is ON for a longer fraction than k_on
            if np.count_nonzero(self.count_mask[i])/self.count_mask[i].size > k_on:
                d_loc = np.gradient(localizations[i], axis=0)   #calculate different between subsequent localizations
                d_loc = np.linalg.norm(d_loc, axis=1)       # (x, y) --> Euklidean norm
                # exclude exp i if there are too many large jumps between localizations (allow one in 100 to account for outliers)
                if np.count_nonzero(d_loc > d_pos)/d_loc.size < 0.01:
                    # remove outlier localizations
                    loc_med = np.median(localizations[i], axis=0)
                    loc_centered = localizations[i] - loc_med
                    d_med = np.sqrt(np.sum(loc_centered**2, axis=1))   # d_med is Euklidean distance from median
                    mdev = np.median(d_med)     # median deviation from median
                    s_med = d_med/mdev      # standardized deviation from median (robust to outliers)
                    k_outlier = 5
                    t_fit = self.t_mask[i][s_med < k_outlier]   # only use localizations within given range around median
                    t_fit = (t_fit - min(t_fit))/(max(t_fit) - min(t_fit))            # when values grow too large fitting doesn't work, so don't take t_mask directly
                    loc_fit = localizations[i][s_med < k_outlier]
                    if loc_fit.size > 0:    # if localizations pass filtering, fit polynomial to presumptive fiducial
                        p_fit.append(np.polyfit(t_fit, loc_fit, k_fit))
                    else:
                        p_fit.append(np.empty(k_fit))
                        warnings.warn(f'exp {i} did not pass outlier test')
                        
        if len(p_fit) == 0:
            warnings.warn('No fiducials found. Could not apply drift correction')
            loc_corr = localizations
            p_fit_avg = p_fit
        else:   # drift correction analogous to subtract_drifts_fit()
            p_fit_avg = np.average(p_fit, axis=0)   #might find more elegant way of averaging if required
            
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
    
    
    
    def _crop_arr(self, arr, x0, y0, size_x, size_y):
        '''
        Crop array to specified region. Values are rounded to next integer.

        Parameters
        ----------
        arr : array to crop
        x0 : starting point on axis 0
        y0 : starting point on axis 1
        size_x : number of entries to keep along axis 0
        size_y : number of entries to keep along axis 1

        Returns
        -------
        Cropped array.
        '''
        x0 = int(round(x0))
        y0 = int(round(y0))
        size_x = int(round(size_x))
        size_y = int(round(size_y))
        return arr[x0:x0+size_x, y0:y0+size_y]
    
    
    def _p_correct_background(self, p, sbr):
        '''
        Correct p-parameters for background following formula (S30) from Balzarotti et al., Science (2017).

        Parameters
        ----------
        p : grid of p-parameters
        sbr : signal-to-background ratio

        Returns
        -------
        Corrected p-parameters.
        '''
        # formula (S30) from Balzarotti et al., Science (2017).
        # compatible with LMS: p is 2D array, sbr is 1D array; and MLE: p is 3D array, sbr is scalar 
        p = sbr/(sbr + 1) * p + 1/(1 + sbr) * 1/self.parameters.n_tcp
        return p
  
 


