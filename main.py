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
 



def load_data(n_read=-1):
    t1 = time.time()
    mfxdata = DataMinflux(mfxparam, n_read=n_read)
    # mfxdata.find_offset()
    mfxdata.create_grid()
    mfxdata.create_timestamps()
    t2 = time.time()
    print('data creation time: ', t2 - t1)
    
    return mfxdata
    
   
def localization(psf=None):
    
    t1 = time.time()
    
    mfxloc = MinfluxLocalization2D(mfxparam, mfxdata, psf_data=psf)
    mfxloc.remove_activation_counts()
    mfxloc.bin_counts(k_bin=mfxparam.k_bin)
    mfxloc.filter_counts(mfxparam.filter_type, mfxparam.t_filter)
    
    mfxloc.threshold_counts(mfxparam.count_threshold, t_min=mfxparam.t_threshold, counts=None,
                          bin_var=mfxparam.t_filter+1, thresh_var=mfxparam.variation_threshold, 
                          cfr_max=mfxparam.cfr_max, subtract_background=mfxparam.subtract_bg)

    if mfxparam.n_photon == -1:
        loc = mfxloc.estimate_position(counts_thresh=mfxloc.counts_thresh, estimator=mfxparam.estimator)
    else:
        # counts_bg_subtracted = [c - bg for c, bg in zip(mfxloc.counts_raw, mfxloc.counts_background)]
        events_binned = mfxloc.bin_emission_events(counts=mfxloc.counts_raw, n_bin=mfxparam.n_photon)
        loc = mfxloc.estimate_position(counts_thresh=events_binned, estimator=mfxparam.estimator)
        
    # loc = [loc_exp[::30] for loc_exp in loc]
    # loc = mfxloc.crop_localizations(-10, -3, -7, 0, localizations=loc)
    
    if mfxparam.subtract_drifts != -1: loc_corr, p_drift = mfxloc.subtract_drifts_fit(k_fit=mfxparam.subtract_drifts, fit_separate=True)
    else: loc_corr, p_drift = None, None
    
    mfxdata.get_localization_data(mfxloc)
    
    t2 = time.time()
    print('localization time: ', t2 - t1)
    
    return mfxloc
    
    
def visualization(save_plots=False):
    t1 = time.time()
    
    mfxvis = MinfluxVisualization2D(mfxparam, mfxdata, plot_style='default', 
                                    save_plots=save_plots)
    
    # mfxvis.plot_count_traces(counts=mfxdata.counts_processed)
    # mfxvis.plot_count_histogram(d_bins=1, log=True)
    
    mfxvis.plot_localizations_scatter(localizations=mfxdata.localizations, show_lines=True, color_code='time')
    # mfxvis.plot_localizations_histogram(localizations=mfxdata.localizations, 
    #                                   px_size = 0.5, shift_hist=True)
    # mfxvis.plot_localizations_gauss(localizations=mfxdata.localizations, 
    #                                 sigma=0.5, px_size = 0.2)
    if hasattr(mfxloc, 'p_drift'):
        p_drift = mfxloc.p_drift
    else:
        p_drift = None
    # mfxvis.plot_localization_traces(localizations=mfxdata.localizations_raw, p_drift=p_drift, centering=False)
    
    t2 = time.time()
    print('visualization time: ', t2 - t1)
    
    return mfxvis

    
def analysis(save_plots=False):
    t1 = time.time()
    
    mfxanalysis = MinfluxAnalysisBeadSample(mfxparam, mfxdata, mfxloc, t_crop=None,
                                            save_plots=save_plots)
    mfxanalysis.fft_counts(average_exp=True, sum_counts=False, log_y=False, f_lim=[0.1, None])   
   
    # mfxanalysis.precision_series([2000, 5000, 500], average_exp=True, exponent=None, plot_localization_series=False)
    
    
    
    t2 = time.time()
    print('analysis time: ', t2 - t1)
    
    return mfxanalysis





def main():
    pass



#%%
if __name__ == '__main__':
    main()
    sample_type = 'bead'
    save_plots = '.pdf'
    
    mfxparam = MinfluxParameters(sample_type)
    
    #%% 
    mfxdata = load_data()
    
    #%%
    psf_model = PSFmodel('doughnut', mfxparam)
    psf_model.create_psf_model(300, 0.5)
    
    mfxloc = localization(psf=psf_model)
    
    #%%
    mfxvis = visualization(save_plots=save_plots)
    
    #%%
    # mfxanalysis = analysis(save_plots=save_plots) 
    
    if save_plots:
        mfxparam.save_all()
    
    
   