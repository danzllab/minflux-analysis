# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:54:47 2023

@author: jvorlauf_admin
"""

import numpy as np
import time

from minflux_parameters import MinfluxParameters
from minflux_dataIO import DataMinflux
from minflux_psf import PSFmodel
from minflux_localization import MinfluxLocalization2D
from minflux_visualization import MinfluxVisualization2D
from minflux_analysis import MinfluxAnalysisBeadSample, MinfluxAnalysisBlinkingSample
 



def load_data(mfxparam, n_read=-1):
    t1 = time.time()
    mfxdata = DataMinflux(mfxparam, n_read=n_read)
    # mfxdata.find_offset()     # find offset corresponding to metadata to discard before reading counts; run if raw data format changes
    mfxdata.create_grid()   # create grid of positions of the tip/tilt mirror from given parameters and assign counts to positions/experiments
    mfxdata.create_timestamps()     # create timestamp for every set of photon counts
    t2 = time.time()
    print('data creation time: ', t2 - t1)
    
    return mfxdata
    
   
def localization(mfxparam, mfxdata, psf=None, plot_mle=False):
    
    t1 = time.time()
    
    mfxloc = MinfluxLocalization2D(mfxparam, mfxdata, psf_data=psf)
    mfxloc.remove_activation_counts()   # remove counts when activation was active based on changed data format
    mfxloc.bin_counts(k_bin=mfxparam.k_bin)     # bin k_bin subsequent photon counts
    mfxloc.filter_counts(mfxparam.filter_type, mfxparam.t_filter)   #filter counts to remove noise
    
    # threshold count traces to extract single-molecule emission events; based on count intensities (lower + upper threshold), on-time, local variance, center frequency ratio (useful for iterative minflux)
    mfxloc.threshold_counts(mfxparam.count_threshold, t_min=mfxparam.t_threshold, counts=None,
                          bin_var=mfxparam.t_filter+1, thresh_var=mfxparam.variation_threshold, 
                          cfr_max=mfxparam.cfr_max, subtract_background=mfxparam.subtract_bg)   # subtract_background not recommended - background should be modeled, not subtracted!

    if mfxparam.n_photon < 0:   # if photon bin parameter negative, directly use thresholded, filtered counts 
        loc = mfxloc.estimate_position(counts_thresh=mfxloc.counts_thresh, estimator=mfxparam.estimator)
    else:   # if photon bin parameter positive, use it to bin raw counts where emission event was detected
        events_binned = mfxloc.bin_emission_events(counts=mfxloc.counts_raw, n_bin=mfxparam.n_photon)
        loc = mfxloc.estimate_position(counts_thresh=events_binned, estimator=mfxparam.estimator, estimator_param={'plot_mle': plot_mle})
        
    # crude filtering of localizations
    # loc = [loc_exp[::30] for loc_exp in loc]  # for faster plotting
    # loc = mfxloc.crop_localizations(-10, -3, -7, 0, localizations=loc)    #crop localizations
    
    if mfxparam.subtract_drifts >= 0:   # if drift subtraction parameter positive or 0, apply post host drift correction using polynomial fit
        loc_corr, p_drift = mfxloc.subtract_drifts_fit(k_fit=mfxparam.subtract_drifts, fit_separate=True)
    else: 
        loc_corr, p_drift = None, None
    
    mfxdata.get_localization_data(mfxloc)
    
    t2 = time.time()
    print('localization time: ', t2 - t1)
    
    return mfxloc
    
    
def visualization(mfxparam, mfxdata, save_plots=False, plot_types=['gauss', 'count_traces', 'scatter_tile']):
    '''
    implemented plot_types: count_traces, count_histogram, scatter_tile, scatter_time, localization_histogram, gauss, localization_traces
    '''
    
    t1 = time.time()
    
    mfxvis = MinfluxVisualization2D(mfxparam, mfxdata, plot_style='default', 
                                    save_plots=save_plots)
    
    if 'count_traces' in plot_types:    # plot thresholded and filtered counts against time
        mfxvis.plot_count_traces(counts=mfxdata.counts_processed, show_trace_segmentation=True)   
    if 'count_histogram' in plot_types:     # plot histogram of processed counts
        mfxvis.plot_count_histogram(d_bins=1, log=True)     
    
    if 'scatter_tile' in plot_types:     # scatter-plot, color-coded based on tile (i.e., position of the tip/tilt mirror)
        mfxvis.plot_localizations_scatter(localizations=mfxdata.localizations, show_lines=False, color_code='tile')
    if 'scatter_time' in plot_types:     # scatter-plot, color-coded based on time (i.e., position of the tip/tilt mirror)
        mfxvis.plot_localizations_scatter(localizations=mfxdata.localizations, show_lines=False, color_code='time')
    if 'localization_histogram' in plot_types:  # 2D histogram displayed as heatmap; shift-hist option for straightforward smoothing of points
        mfxvis.plot_localizations_histogram(localizations=mfxdata.localizations, 
                                            px_size=0.5, shift_hist=False)
    if 'gauss' in plot_types:   # plot localizations as Gaussians of width sigma (either single value or list containing one per localization)
        mfxvis.plot_localizations_gauss(localizations=mfxdata.localizations, 
                                        sigma=1, px_size=0.2)
    
    if hasattr(mfxloc, 'p_drift'):  # if they exist, use polynomial coefficients from post hoc drift correction 
        p_drift = mfxloc.p_drift
    else:
        p_drift = None
    if 'localization_traces' in plot_types:     # plot localization coordinates against time
        mfxvis.plot_localization_traces(localizations=mfxdata.localizations_raw, p_drift=p_drift, centering=False)
    
    t2 = time.time()
    print('visualization time: ', t2 - t1)
    
    return mfxvis

    
def analysis(mfxparam, mfxdata, mfxloc, save_plots=False):
    t1 = time.time()
    
    mfxanalysis = MinfluxAnalysisBeadSample(mfxparam, mfxdata, mfxloc, t_crop=None,
                                            save_plots=save_plots)
    mfxanalysis.fft_counts(average_exp=True, sum_counts=False, log_y=False, f_lim=[0.1, None])   
   
    # mfxanalysis.precision_series([2000, 5000, 500], average_exp=True, exponent=None, plot_localization_series=False)
    
    
    
    t2 = time.time()
    print('analysis time: ', t2 - t1)
    
    return mfxanalysis




#%% functions executed here
if __name__ == '__main__':
    sample_type = 'blink'   # sets parameter profile
    save_plots = '.png'     # file ending or None if don't want to save
    
    mfxparam = MinfluxParameters(sample_type)   #load parameter profile
    
    #%% load minflux counts from file
    mfxdata = load_data(mfxparam)   # optional integer parameter determined how many iterations to load (-1: entire file)
    
    #%% minflux localization
    psf_model = PSFmodel('doughnut', mfxparam)  # initialize PSF-model (optionally, use experimental calibration instead)
    psf_model.create_psf_model(300, 0.2)    # grid size, pixel size
    
    mfxloc = localization(mfxparam, mfxdata, psf=psf_model, plot_mle=False)
    
    #%% data visualization
    mfxvis = visualization(mfxparam, mfxdata, save_plots=save_plots, plot_types=['localization_traces', 'count_histogram', 'gauss'])
    
    #%% data analysis
    # mfxanalysis = analysis(mfxparam, mfxdata, mfxloc, save_plots=save_plots) 
    
    # if plots are saved, also save parameters to .txt file
    if save_plots:
        mfxparam.save_all()
    
    
   