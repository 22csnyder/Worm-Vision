# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:33:34 2015

@author: csnyder
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:19:20 2015

@author: csnyder
"""

import numpy as np
#from WormBox.BaseClasses import tEnvironment as Env
from WormBox.VisClasses import MultiChannelWorm
from WormScene import WormScene
from Display import Display



stop do not run this yet its not ready

dataset='151111_Vol_Agarpad_W2_y334'

vol24_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151111_1457_Vol_Agarpad_W2_y334_z24/TIFF/'#Starts on 10, skip every first file of the 32
vol32_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151111_1459_Vol_Agarpad_W2_y334_z32/TIFF/'#Starts on 16, skip first frame
vol40_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151111_1500_Vol_Agarpad_W2_y334_z40/TIFF/'#Starts on 24, skip first frame
vol48_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151111_1502_Vol_Agarpad_W2_y334_z48/TIFF/'#Starts on 31, skip first frame

cache_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151111_Vol_Agarpad_W2_y334/Cache/'#must end in '/' for now
save_results_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151111_Vol_Agarpad_W2_y334/Results/'

cm_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151111_CM_Agarpad_W2_HiRes_Ref/TIFF'

um_window_shape=np.array([10.0,10.0,10.0])
nm_window_shape=1000*um_window_shape

#A worm can have multiple data channels, but only 1 set of neuron segments
worm=MultiChannelWorm(nm_window_shape,cm_dir)
worm.cache_dir=cache_dir
worm.results_dir=save_results_dir

################################################
#worm.segments_filename='Segments_cm_z32.pkl'#check lower
#worm.segments_filename='NonSave'
################################################


##Load data into channels
worm.add_channel('cm',cm_dir)#Hi Res 3D
worm.add_channel('bad_vol24',vol24_dir)
worm.add_channel('bad_vol32',vol32_dir)#first frame at start of each vol is bad
worm.add_channel('bad_vol40',vol40_dir)
worm.add_channel('bad_vol48',vol48_dir)

#def compute_median(array):
#    s=array.shape[1]//2
#    A=np.median(array[:,:s],axis=0)#Split up computation for memory considerations
#    B=np.median(array[:,s:],axis=0)
#    return np.vstack([A,B])





##Create channel derived from other data
#throw out first plane of timepoint and compute median(array):
def mirror_first_and_median(array):
    s=array.shape[1]//2
    A=np.median(array[:,:s],axis=0)#Split up computation for memory considerations
    B=np.median(array[:,s:],axis=0)
    C=np.vstack([A,B])
    C[0,:,:]=C[1,:,:]#discard first frame without messing up spacing
    return C

#Redo process

worm.add_channel_derived_from('bad_vol24',with_process=mirror_first_and_median, named='med24')
worm.add_channel_derived_from('bad_vol32',with_process=mirror_first_and_median, named='med32')
worm.add_channel_derived_from('bad_vol40',with_process=mirror_first_and_median, named='med40')
worm.add_channel_derived_from('bad_vol48',with_process=mirror_first_and_median, named='med48')


#worm.add_channel('cm_debug',cm_dir)

gui=Display()
gui.worm=worm

gui.worm.center.set_value([17920,67670,435267])#nm
gui.worm.radius.set_value([10200,12316,13757])#nm

##############Comment/Uncomment Different Datasets################

#worm.segments_filename='Segments_cm.pkl'
#gui.add_scene('Whole_Worm_High_Res',WormScene(downsample=5))
#gui.add_scene('Zoom_High_Res',WormScene(is_local_view=True))
#gui.hash_volume=dict({
#    'Whole_Worm_High_Res':'cm',
#    'Zoom_High_Res':'cm',})

#worm.segments_filename='Segments_z24.pkl'
#gui.add_scene('Whole_Worm_z24', WormScene(downsample=4) )
#gui.add_scene('Zoom_Med_time_series_z24', WormScene(is_local_view=True) )
#gui.hash_volume=dict({
#    'Whole_Worm_z24':'med24',
#    'Zoom_Med_time_series_z24':'med24',})

#worm.segments_filename='Segments_z32.pkl'
#gui.add_scene('Whole_Worm_z32', WormScene(downsample=4) )
#gui.add_scene('Zoom_Med_time_series_z32', WormScene(is_local_view=True) )
#gui.hash_volume=dict({
#    'Whole_Worm_z32':'med32',
#    'Zoom_Med_time_series_z32':'med32',})


#worm.segments_filename='Segments_z40.pkl'
#gui.add_scene('Whole_Worm_z40', WormScene(downsample=4) )
#gui.add_scene('Zoom_Med_time_series_z40', WormScene(is_local_view=True) )
#gui.hash_volume=dict({
#    'Whole_Worm_z40':'med40',
#    'Zoom_Med_time_series_z40':'med40',})

#worm.segments_filename='Segments_z48.pkl'
#gui.add_scene('Whole_Worm_z48', WormScene(downsample=4) )
#gui.add_scene('Zoom_Med_time_series_z48', WormScene(is_local_view=True) )
#gui.hash_volume=dict({
#    'Whole_Worm_z48':'med48',
#    'Zoom_Med_time_series_z48':'med48',})
    

#####Compare all at once (qualitative)#####
#gui.add_scene('Whole_Worm_High_Res',WormScene(downsample=5))
#gui.add_scene('Zoom_High_Res',WormScene(is_local_view=True))
#gui.add_scene('Zoom_Med_time_series_z24', WormScene(is_local_view=True) )
#gui.add_scene('Zoom_Med_time_series_z32', WormScene(is_local_view=True) )
#gui.add_scene('Zoom_Med_time_series_z40', WormScene(is_local_view=True) )
#gui.add_scene('Zoom_Med_time_series_z48', WormScene(is_local_view=True) )
#gui.hash_volume=dict({
#    'Whole_Worm_High_Res':'cm',
#    'Zoom_High_Res':'cm',
#    'Zoom_Med_time_series_z24':'med24',
#    'Zoom_Med_time_series_z32':'med32',
#    'Zoom_Med_time_series_z40':'med40',
#    'Zoom_Med_time_series_z48':'med48',
#    })


gui.start()

#display.configure_traits()

import dill
fs24='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151111_Vol_Agarpad_W1-1_y334/Results/Segments_z24.pkl'
with open(fs24,'rb') as handle:
    s24=dill.load(handle)
fs32='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151111_Vol_Agarpad_W1-1_y334/Results/Segments_z32.pkl'
with open(fs32,'rb') as handle:
    s32=dill.load(handle)
fs40='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151111_Vol_Agarpad_W1-1_y334/Results/Segments_z40.pkl'
with open(fs40,'rb') as handle:
    s40=dill.load(handle)
fs48='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151111_Vol_Agarpad_W1-1_y334/Results/Segments_z48.pkl'
with open(fs48,'rb') as handle:
    s48=dill.load(handle)





