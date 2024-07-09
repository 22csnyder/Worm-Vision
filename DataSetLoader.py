# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:18:04 2015

@author: csnyder
"""

#class DatasetLoader:
#    
#    def __init__(self):
#        pass
#    
#    def __call__(dataset

from WormBox.VisClasses import VisWorm

import numpy as np

def load_data(name):
    
    if name == '151005_Glycerol_3ch_Dev_II_W4':
        read_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151005_CM_Glycerol_3ch_Dev_II_W4/1/TIFF'
        write_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151005_Glycerol_3ch_Dev_II_W4/Results'
        
        um_window_shape=np.array([10.0,10.0,10.0])
        
        nm_window_shape=1000*um_window_shape
        worm=VisWorm(nm_window_shape,read_dir,write_dir)
        
        
        