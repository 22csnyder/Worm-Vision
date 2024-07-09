# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:06:44 2015

@author: csnyder
"""

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

##Create channel derived from other data
def compute_median(array):
    s=array.shape[1]//2
    A=np.median(array[:,:s],axis=0)#Split up computation for memory considerations
    B=np.median(array[:,s:],axis=0)
    
    return np.vstack([A,B])
    
dataset='151016_CM_Anesthetic_Dev_II_W3'

cm1_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151016_CM_Anesthetic_Dev_II_W3/1/TIFF'
cm2_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151016_CM_Anesthetic_Dev_II_W3/2/TIFF'
cache_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151016_CM_Anesthetic_Dev_II_W3/Cache'
save_results_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151016_CM_Anesthetic_Dev_II_W3/Results'


um_window_shape=np.array([20.0, 20.0, 20.0])##CAREFUL !!. This is okay since not segmenting in this image
#um_window_shape=np.array([40.0, 40.0,40.0])##CAREFUL !!. This is okay since not segmenting in this image
#um_window_shape=np.array([10.0, 10.0,10.0])##CAREFUL !!. This is okay since not segmenting in this image
nm_window_shape=1000*um_window_shape
#nm_image_shape=np.array([25000,150000,630000]).astype(np.int)

worm=MultiChannelWorm(nm_window_shape,cm1_dir)
worm.cache_dir=cache_dir
worm.results_dir=save_results_dir

##Load data into channels
#worm.add_channel('CM',cm_dir)#Hi Res 3D
worm.add_channel('cm1',cm1_dir)
worm.add_channel('cm2',cm2_dir)


gui=Display()
gui.worm=worm

ds=5#downsample

#'full_CM_scene'
#'local_CM_scene'


gui.add_scene('global_151016_CM1',WormScene(downsample=ds))
gui.add_scene('local_cm1',WormScene(is_local_view=True))
gui.add_scene('local_cm2',WormScene(is_local_view=True))
#gui.add_scene('global_151016_CM2',WormScene(downsample=ds))


####CAREFUL#### to avoid error in neuron placement, never adjust radius...###

#gui.worm.center.set_value([6250,51885,63882])#worm.center is in nm
gui.worm.center.set_value([15960,48765,60165])#worm.center is in nm
#rad=gui.worm.radius.__array__()
#gui.worm.radius.set_value(rad*2)#worm.center is in nm



gui.hash_volume=dict(
    {
    'global_151016_CM1':'cm1',
    'local_cm1':'cm1',
    'local_cm2':'cm2',
#    'global_151016_CM2':'cm2',
     })


#from mayavi import mlab
#sc1=gui.scene_dict['local_cm1']
#sc2=gui.scene_dict['local_cm2']
#mlab.sync_camera(sc1.mayavi_scene,sc2.mayavi_scene)
#mlab.sync_camera(sc1.scene,sc2.scene)

gui.start()

####Pickleing seems to work if I just make sure not to have apo_reader as an attribute
#import os
#import dill
#f=os.path.join(worm.results_dir,worm.segments_filename)
#with open(f,'rb') as handle:
#    s=dill.load(handle)

#from WormBox.BaseClasses import Segments
#d=Segments()
#d['l']=10
#d['j']=['u']
#
#dill.detect.trace(True)
#
#with open(f,'wb') as handle:
#    dill.dump(d,handle)
#
#with open(f,'rb') as handle:
#    s=dill.load(handle)
#
#h=open(f,'rb')


