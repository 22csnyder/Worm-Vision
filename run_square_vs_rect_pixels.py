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

dataset='151016_1801_Vol_no_stimulus_W3'

cm_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151005_CM_Glycerol_3ch_Dev_II_W4/1/TIFF'
ini_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151016_1801_Vol_Anesthetic_Dev_II_no_stimulus_5mins_after_W3'
Vol_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151016_1801_Vol_Anesthetic_Dev_II_no_stimulus_5mins_after_W3/TIFF/'#starts at 19
opt_Vol_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151016_1801_Vol_Anesthetic_Dev_II_no_stimulus_5mins_after_W3/TIFF_opt/'#starts at 19
save_results_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151016_1801_Vol_Anesthetic_W3/Results/'
cache_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151016_1801_Vol_Anesthetic_W3/Cache/'


um_window_shape=np.array([10.0,10.0,10.0])
nm_window_shape=1000*um_window_shape


#A worm can have multiple data channels, but only 1 set of neuron segments
worm=MultiChannelWorm(nm_window_shape,ini_dir)
worm.cache_dir=cache_dir

##Load data into channels
#worm.add_channel('CM',cm_dir)#Hi Res 3D
worm.add_channel('Vol',Vol_dir)#Time varying 4D
worm.add_channel('opt',opt_Vol_dir)#Time varying 4D#double x spacing

##Create channel derived from other data
def compute_median(array):
    s=5
    A=np.median(array[:,:s],axis=0)#Split up computation for memory considerations
    B=np.median(array[:,s:],axis=0)
    
    return np.vstack([A,B])


worm.add_channel_derived_from('Vol',with_process=compute_median, named='Med')
worm.add_channel_derived_from('opt',with_process=compute_median, named='optMed')

#worm['opt'].spacing[0]/=2.0#reduce by half because#Added a .ini file in TIFF and TIFF_opt
#worm['opt'].nm_voxel_shape/=2.0

##Tell the program what image data to use for calculating the segmentation boundaries
#worm.segmentation_channel_is('Med')

display=Display()#initialize display
display.worm=worm#hook worm up to display

##Create Scenes that have properties defining how to display the channels
## "WormScene" defines mouse&keyboard interaction properties
display.add_scene('full_Vol_scene', WormScene() )
display.add_scene('full_opt_scene', WormScene() )



#display.worm.center.set_value([37*300,488*150,344*150])#worm.center is in nm
#rad=display.worm.radius.__array__()
#display.worm.radius.set_value(rad*2)#worm.center is in nm



###Tell which channels should display volume (ImageData) on which scene
display.hash_volume=dict(
    {
    'full_Vol_scene':'Med',
     'full_opt_scene':'optMed',
     })
    

display.configure_traits()