# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:45:02 2015

@author: csnyder
"""
import numpy as np

from WormBox.BaseClasses import tEnvironment as Env
from WormBox.BaseClasses import tWorm
from WormBox.VisClasses import Channel,MultiChannelWorm
from WormScene import WormScene
from Display import Display


ini_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151005_CM_Glycerol_3ch_Dev_II_W4'
save_results_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151005_Glycerol_3ch_Dev_II_W4/Results'


um_window_shape=np.array([10.0,10.0,10.0])
nm_window_shape=1000*um_window_shape

worm=MultiChannelWorm(nm_window_shape,ini_dir)

#worm=tWorm(ini_dir)

#worm.get_ini_file()


cm_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151005_CM_Glycerol_3ch_Dev_II_W4/1/TIFF'
Vol_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151005_2209_Vol_Glycerol_3ch_Dev_II_W4_322L25P/TIFF'


###Need to get starting image too

#WormScene

worm.add_channel('CM',cm_dir)

#worm.channels['CM'].dask_read_data()

worm.add_channel('Vol',Vol_dir)

def compute_median(array):
    return np.median(array,axis=0)



#worm.add_channel_derived_from('Vol',with_process=compute_median, named='Med')





#large_cm_scene=WormScene()
#large_cm_scene.channel_name='CM'
#
#small_view_scene=WormScene()
#small_view_scene.channel_name='CM'
#small_view_scene.is_local_view=True
#
#
#small_median_scene=WormScene()
#small_median_scene.channel_name='Vol'
#small_median_scene.is_local_view=True


#worm.segmentation_channel_is('Med')



#Scenes=[large_cm_scene,s


#MedX=np.load(worm.write_dir+'Median3D.npy')
##V=np.median(worm.daX,axis=0)
##np.save(worm.write_dir+'Median3D.npy',V)
try:
    cm=worm['CM']
    vo=worm['Vol']
except:
    pass




opt_Vol_dir=Vol_dir+'_opt'