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
from WormBox.BaseClasses import tEnvironment as Env
from WormBox.BaseClasses import tWorm
from WormBox.VisClasses import Channel,MultiChannelWorm
from WormScene import WormScene
from Display import Display

from traits.api import HasTraits,Int,Instance,List
from traitsui.api import HGroup,View,VGroup,HGroup,Include,TextEditor,Item,spring,ViewSubElement
from traitsui.api import InstanceEditor

from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from tvtk.pyface.scene_editor import SceneEditor

from mayavi.core.api import Engine

dataset='151005_Glycerol_3ch_Dev_II_W4'

#cm_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151005_CM_Glycerol_3ch_Dev_II_W4/1/TIFF'
#ini_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151016_1801_Vol_Anesthetic_Dev_II_no_stimulus_5mins_after_W3'
#Vol_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151016_1801_Vol_Anesthetic_Dev_II_no_stimulus_5mins_after_W3/TIFF/'#starts at 19
#opt_Vol_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151016_1801_Vol_Anesthetic_Dev_II_no_stimulus_5mins_after_W3/TIFF_opt/'#starts at 19

Vol15_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151005_2204_Vol_Glycerol_3ch_Dev_II_W4/TIFF/'#zps=15,start=23
Vol25_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151005_2209_Vol_Glycerol_3ch_Dev_II_W4_322L25P/TIFF/'#zps=25,start=20
Vol21_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151005_2211_Vol_Glycerol_3ch_Dev_II_W4_380L21P/TIFF/'#zps=21,start=23

cache_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151005_Glycerol_3ch_Dev_II_W4/Cache/'
save_results_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151005_Glycerol_3ch_Dev_II_W4/Results/'#make sure to put slash on end


cm_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151005_CM_Glycerol_3ch_Dev_II_W4/1/TIFF'#for debug
#worm=MultiChannelWorm(nm_window_shape,cm_dir)


um_window_shape=np.array([10.0,10.0,10.0])
nm_window_shape=1000*um_window_shape
nm_image_shape=np.array([25000,150000,630000]).astype(np.int)

#A worm can have multiple data channels, but only 1 set of neuron segments
#worm=MultiChannelWorm(nm_window_shape,ini_dir)
#worm=MultiChannelWorm(nm_window_shape,nm_image_shape)
worm=MultiChannelWorm(nm_window_shape,Vol21_dir)
worm.cache_dir=cache_dir
worm.results_dir=save_results_dir

##Load data into channels
#worm.add_channel('CM',cm_dir)#Hi Res 3D
worm.add_channel('Vol15',Vol15_dir)
worm.add_channel('Vol21',Vol21_dir)
worm.add_channel('Vol25',Vol25_dir)

##Create channel derived from other data
def compute_median(array):
    s=array.shape[1]//2
    A=np.median(array[:,:s],axis=0)#Split up computation for memory considerations
    B=np.median(array[:,s:],axis=0)
    
    return np.vstack([A,B])

#worm.add_channel_derived_from('Vol21',with_process=compute_median, named='Med21')


#worm.add_channel('cm_debug',cm_dir)

#gui=Display()
#gui.worm=worm

#gui.add_scene('full_Med21_scene', WormScene(downsample=5) )
#gui.add_scene('local_Med21_scene', WormScene(is_local_view=True) )
#
#gui.hash_volume=dict(
#    {
#    'full_Med21_scene':'Med21',
#    'local_Med21_scene':'Med21',
#     })

#gui.hash_volume=dict(
#    {
#    'full_Med21_scene':'cm_debug',
#    'local_Med21_scene':'cm_debug',
#     })

worm.add_channel_derived_from('Vol15',with_process=compute_median, named='Med15')
worm.add_channel_derived_from('Vol21',with_process=compute_median, named='Med21')
worm.add_channel_derived_from('Vol25',with_process=compute_median, named='Med25')

##Tell the program what image data to use for calculating the segmentation boundaries
#worm.segmentation_channel_is('Med')

gui=Display()
gui.worm=worm


gui.add_scene('full_Med21_scene', WormScene() )
gui.add_scene('local_Med15_scene', WormScene(is_local_view=True) )
gui.add_scene('local_Med21_scene', WormScene(is_local_view=True) )
gui.add_scene('local_Med25_scene', WormScene(is_local_view=True) )

#gui.worm.center.set_value([6250,51885,63882])#worm.center is in nm
gui.worm.center.set_value([6250,64290,57267])#worm.center is in nm
rad=gui.worm.radius.__array__()
gui.worm.radius.set_value(rad*2)#worm.center is in nm


ch15=worm['Med15']
ch21=worm['Med21']
ch25=worm['Med25']

#gui.sync_trait('volume_control',ch15,'small_vol_module')
#ch15.sync_trait('small_vol_module',ch21)


gui.hash_volume=dict(
    {
    'full_Med21_scene':'Med21',
    'local_Med21_scene':'Med21',
    'local_Med15_scene':'Med15',
    'local_Med25_scene':'Med25',
     })



gui.start()

#display.configure_traits()