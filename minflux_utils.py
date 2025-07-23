# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:39:24 2023

@author: jvorlauf_admin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import pyximport; pyximport.install()
# from cython_utils import _nudft_cython

def _nudft(t, y, f):     # non-uniform discrete Fourier Transform Type 2
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


def _sort_time_coded_data(t, data):
    t_flat = np.hstack(t)
    if data[0].ndim == 1:
        data_flat = np.hstack(data)
    else:
        data_flat = np.vstack(data)    
    sort_ind = np.argsort(t_flat)
    t_sorted = t_flat[sort_ind]
    data_sorted = data_flat[sort_ind]
    return t_sorted, data_sorted
    

def _sum_counts(counts):
    counts_sum = [np.sum(counts_exp, axis=1) for counts_exp in counts]
    return counts_sum
    
    
def _f_gauss(fwhm, grid_size, px_size):
    coord = np.arange(-grid_size/2, grid_size/2, px_size)
    xx, yy = np.meshgrid(coord, coord)
    rr2 = xx**2 + yy**2
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    pixel_intensities = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-rr2/(2*sigma**2))
    return pixel_intensities


def _f_doughnut(fwhm, grid_size, px_size):
    coord = np.arange(-grid_size/2, grid_size/2, px_size)
    xx, yy = np.meshgrid(coord, coord)
    rr2 = xx**2 + yy**2
    f = rr2/fwhm**2 * np.exp(-4*np.log(2)*rr2/fwhm**2)  # values not normalized
    return f


def _draw_localizations_scatter(fig, ax, localizations, t, show_lines=False, color_code='time'):
    t, loc_all = _sort_time_coded_data(t, localizations)
    
    ax.set_aspect('equal')
    
    if show_lines: ax.plot(loc_all[:,0], loc_all[:,1], color='k', alpha=0.3, linewidth=0.5, zorder=1)
    if color_code == 'time':
        h = ax.scatter(loc_all[:,0], loc_all[:,1], c=t, cmap='cividis',
                       alpha=1, s=5, linewidth=0.5, zorder=2)
        divider = make_axes_locatable(ax)   # this and next line needed for colorbar to match plot height
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = fig.colorbar(h, cax=cax)
        cbar.solids.set(alpha=1)
        cax.set_ylabel('Time [s]')
    elif color_code == 'tiles':
        for i in range(len(localizations)):
            if localizations[i].size > 0:
                ax.scatter(localizations[i][:,0], localizations[i][:,1], zorder=2,
                           facecolors='none', edgecolors='C'+str(i), alpha = 0.5, s=5, linewidth=0.5)
    else:
        ax.scatter(loc_all[:,0], loc_all[:,1], alpha=0.5, s=5, linewidth=0.5, zorder=2)

    ax.set_aspect('equal')


# =============================================================================
# TODO implement properly if desired
# def _draw_localizations_histogram(fig, ax, localizations, vlim=None, shift_hist=False):
#     loc_all = np.vstack(localizations)
#     
#     n_bins = [round((max(loc_all[:,0]) - min(loc_all[:,0]))/px_size),
#               round((max(loc_all[:,1]) - min(loc_all[:,1]))/px_size)]
#     
#     # define range to get rid of rounding error
#     xyrange = [[min(loc_all[:,0]), min(loc_all[:,0]) + (n_bins[0] - 2)*px_size],
#                [min(loc_all[:,1]), min(loc_all[:,1]) + (n_bins[1] - 2)*px_size]]
#     
#     ax.set_aspect('equal')
#     
#     h = ax.hist2d(loc_all[:,0], loc_all[:,1], n_bins, range=xyrange, cmap='inferno')
#     h_img = h[3]
#     # currently hist is always shifted by 1 to left and right, could adapt for more px?
#     if shift_hist:
#         h_shift = h[0][1:-1,1:-1] + 1/4*(h[0][0:-2,1:-1] + h[0][2:,1:-1] + h[0][1:-1,0:-2] + h[0][1:-1,2:])
#         h_shift *= 0.5
#         ax.clear()
#         h_img = ax.imshow(h_shift.transpose()[::-1,:], extent=[h[1][0], h[1][-1], h[2][0], h[2][-1]],
#                          cmap='inferno')
#     
#     if vlim:
#         h_img.set_clim(vmin=vlim[0], vmax=vlim[1])
#         
#     divider = make_axes_locatable(ax)   # this and next line needed for colorbar to match plot height
#     cax = divider.append_axes("right", size="5%", pad=0.15)
#     fig.colorbar(h_img, cax=cax)
# 
# 
# def _draw_localizations_gauss(fig, ax, localizations, px_size=0.1, sigma=1, vlim=None):
#     fwhm = 2.355 * sigma
#     
#     n_bins = [round((max(loc_all[:,0]) - min(loc_all[:,0]))/px_size),
#               round((max(loc_all[:,1]) - min(loc_all[:,1]))/px_size)]
#     
#     # define range to get rid of rounding error
#     xyrange = [[min(loc_all[:,0]), min(loc_all[:,0]) + (n_bins[0] - 2)*px_size],
#                [min(loc_all[:,1]), min(loc_all[:,1]) + (n_bins[1] - 2)*px_size]]
#     
#     ax.set_aspect('equal')
#     
#     if type(fwhm) == float:
#         h = ax.hist2d(loc_all[:,0], loc_all[:,1], n_bins, range=xyrange, cmap='inferno')
#         h_img = h[3]
#         
#         h_gauss = fftconvolve(h[0], _f_gauss(fwhm=fwhm, grid_size=5*fwhm, px_size=px_size), mode='same')
#         ax.clear()
#         h_img = ax.imshow(h_gauss.transpose()[::-1,:], extent=[h[1][0], h[1][-1], h[2][0], h[2][-1]],
#                          cmap='inferno')
#         
#     else:
#         h = np.zeros(n_bins)
#         for loc, fw in zip(loc_all, fwhm):
#             h += f_gauss(fwhm=fw, center=loc, intensity=1 , n_grid=n_bins)
#     
#     if vlim is not None:
#         h_img.set_clim(vmin=vlim[0], vmax=vlim[1])
#         
#     divider = make_axes_locatable(ax)   # this and next line needed for colorbar to match plot height
#     cax = divider.append_axes("right", size="5%", pad=0.15)
#     fig.colorbar(h_img, cax=cax)
#     
# =============================================================================


if __name__ == '__main__':
    img = _f_gauss(300, [500, 700], 1)
    plt.imshow(img)    
    
    
    
