# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:35:44 2015

Shamelssly lifted from

https://github.com/vortex-exoplanet/VIP/blob/master/vip/stats/clip_sigma.py

"""
import numpy as np

def sigma_filter(frame_tmp, bpix_map, neighbor_box=3, min_neighbors=3, verbose=False):
    """Sigma filtering of pixels in a 2d array.
    
    Parameters
    ----------
    frame_tmp : array_like 
        Input 2d array, image.
    bpix_map: array_like
        Input array of the same size as frame_tmp, indicating the locations of 
        bad/nan pixels by 1 (the rest of the array is set to 0)
    neighbor_box : int, optional
        The side of the square window around each pixel where the sigma and 
        median are calculated.
    min_neighbors : int, optional
        Minimum number of good neighboring pixels to be able to correct the 
        bad/nan pixels
        
    Returns
    -------
    frame_corr : array_like
        Output array with corrected bad/nan pixels
    
    """
    
    if not frame_tmp.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array')

    sz_y = frame_tmp.shape[0]                 # get image y-dim
    sz_x = frame_tmp.shape[1]                 # get image x-dim
    bp = bpix_map                             # temporary bad pixel map
    im = frame_tmp                            # corrected image
    nb = int(np.sum(bpix_map))                  # number of bad pixels remaining
    #In each iteration, correct only those bad pix with sufficient good 'neighbors'
    nit = 0                                 # number of iterations
    while nb > 0:
        nit +=1
        wb = np.where(bp)                   # find bad pixels
        gp = 1 - bp                         # temporary good pixel map
        for n in range(nb):
            #0/ Determine the box around each pixel
            half_box = np.floor(neighbor_box/2.)
            hbox_b = min(half_box,wb[0][n])           # half size of the box at the bottom of the pixel
            hbox_t = min(half_box,sz_y-1-wb[0][n])    # half size of the box at the top of the pixel
            hbox_l = min(half_box,wb[1][n])           # half size of the box to the left of the pixel
            hbox_r = min(half_box,sz_x-1-wb[1][n])    # half size of the box to the right of the pixel
            # but in case we are at an edge, we want to extend the box by one row/column of pixels in the direction opposite to the edge to have 9 px instead of 6: 
            if half_box == 1:
                if wb[0][n] == sz_y-1: hbox_b = hbox_b+1 
                elif wb[0][n] == 0: hbox_t = hbox_t+1
                if wb[1][n] == sz_x-1:hbox_l = hbox_l+1 
                elif wb[1][n] == 0: hbox_r = hbox_r+1

            sgp = gp[(wb[0][n]-hbox_b):(wb[0][n]+hbox_t+1),(wb[1][n]-hbox_l):(wb[1][n]+hbox_r+1)]#!!! The 2nd slicing index refers to 1st element you DONT include in slicing, hence the "+1".
            if int(np.sum(sgp)) >= min_neighbors:
                sim = im[(wb[0][n]-hbox_b):(wb[0][n]+hbox_t+1),(wb[1][n]-hbox_l):(wb[1][n]+hbox_r+1)]
                im[wb[0][n],wb[1][n]] = np.median(sim[np.where(sgp)])
                bp[wb[0][n],wb[1][n]] = 0
        nb = int(np.sum(bp))
    if verbose == True:
        print 'Required number of iterations in the sigma filter: ', nit
    return im
