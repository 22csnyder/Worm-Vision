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

from traits.api import HasTraits,Int,Instance,List
from traitsui.api import HGroup,View,VGroup,HGroup,Include,TextEditor,Item,spring,ViewSubElement
from traitsui.api import InstanceEditor

from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from tvtk.pyface.scene_editor import SceneEditor

from mayavi.core.api import Engine

dataset='151005_Glycerol_3ch_Dev_II_W4'
ini_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151005_CM_Glycerol_3ch_Dev_II_W4'
save_results_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151005_Glycerol_3ch_Dev_II_W4/Results'
cache_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151005_Glycerol_3ch_Dev_II_W4/Cache/'
cm_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151005_CM_Glycerol_3ch_Dev_II_W4/1/TIFF'
Vol_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151005_2209_Vol_Glycerol_3ch_Dev_II_W4_322L25P/TIFF'#starts at 20
um_window_shape=np.array([10.0,10.0,10.0])
nm_window_shape=1000*um_window_shape

#A worm can have multiple data channels, but only 1 set of neuron segments
worm=MultiChannelWorm(nm_window_shape,ini_dir)
worm.cache_dir=cache_dir

##Load data into channels
worm.add_channel('CM',cm_dir)#Hi Res 3D
worm.add_channel('Vol',Vol_dir)#Time varying 4D

##Create channel derived from other data
def compute_median(array):
    return np.median(array,axis=0)
#    med=np.median(array,axis=0)
#    if dataset=='151005_Glycerol_3ch_Dev_II_W4':
#        return med[::-1,:,:]#Flip depending on dataset
#    else:
#        return med

worm.add_channel_derived_from('Vol',with_process=compute_median, named='Med')
#worm.channels['Med'].is_inverted=True


##Tell the program what image data to use for calculating the segmentation boundaries
worm.segmentation_channel_is('Med')

gui=Display()#initialize display
gui.worm=worm#hook worm up to display

##Create Scenes that have properties defining how to display the channels
## "WormScene" defines mouse&keyboard interaction properties
#scene names must not have spaces and must not start with number
#gui.add_scene('full_CM_scene', WormScene(downsample=5) )
#gui.add_scene('local_CM_scene', WormScene(is_local_view=True) )
gui.add_scene('local_median_scene',WormScene(is_local_view=True))
gui.add_scene('full_median_scene',WormScene(downsample=1))


gui.worm.center.set_value([37*300,488*150,344*150])#worm.center is in nm
#rad=gui.worm.radius.__array__()
#gui.worm.radius.set_value(rad*2)#worm.center is in nm



###Tell which channels should display volume (ImageData) on which scene
gui.hash_volume=dict(
    {
#    'full_CM_scene':'CM',
#     'local_CM_scene':'CM',
     'local_median_scene':'Med',
     'full_median_scene':'Med',
     })
    
#from mayavi import mlab
#scn=gui.scene_dict['local_CM_scene']
#mlab.text(0.02,0.95,scn.name,figure=scn.mayavi_scene)

#gui.configure_traits()
gui.start()

#gui.scene_dict['local_CM_scene'].engine
#gui.scene_dict['local_median_scene'].engine


#A "grid" is a small cube that shows where in the
     #...global view the local views are looking
###This describes which channels 
#gui.has_grid







#worm['CM'].gui_on_large_scene(large_cm_scene)




#r.scene_0=Item('smallcm',style='custom')
#r.scene_0=Item('smallcm',style='custom')
#r.scene_0=small_cm_scene
#r.scene_1=small_median_scene




       
#    x=Int(22)  
#    items_sl=List
#    name=List(['1','2','3'])
#    group=Instance(HGroup,('x','x'))
#    def __init__(self):
#        pass
#        self.Scenes=Scenes
#        self.mylist=SceneList(self.Scenes)

#    traits_view=View(
#        Item('mylist',style='custom')
##        Item(name='captain',editor=InstanceEditor(name='mylist',editable=True),style='custom')
#        )
        
        
#        self.items_sl=[Item('scene_'+str(i),style='custom',show_label=False) for i in range(len(self.Scenes))]
#        self.scene_group=HGroup(self.items_sl)
#    custom_view=View(
#
#    custom_view=View(
#        HGroup(
#        'x',
#        'mylist',
##        Include('group')
##        (Item('scene_0',style='custom',show_label=False),
##        Item('scene_1',style='custom',show_label=False),
##        Item('scene_2',style='custom',show_label=False)),
#    #    Item('object.scene_2',style='custom',show_label=False),
#        )
#    )
    
#    traits_view=View(Include('scene_group'))
    
#vis=testVis(Scenes)
#vis=testVis(mylist=Scenes)
    
    
#vis=testVis()
#vis.mylist=Scenes

#from gui import RawDisplay


#custom_item=Item(style='custom')

    

    
chan=gui.worm['Med']    
m=gui.current_neuron['marker']
s=chan.nm_voxel_shape

t=(m//s)*s






#class testVis(HasTraits):
#  
#    mylist=List(WormScene) 



#r.scene_0.traits_view.kind='live'

#myview=View(Item('scene_0',style='custom'))


#r.configure_traits(view=myview,context={'smallcm':small_cm_scene})
#r.configure_traits(view=myview)


#r.configure_traits()



#from traitsui.api import CustomEditor

#l=List(trait=ViewSubElement,value=vis.items_sl)

#vis.scene_group.content=l




#vis.configure_traits()


#vis.scene_1=large_cm_scene
#vis.scene_2=small_cm_scene


#vis.configure_traits(view=custom_view)
#vis.configure_traits()


#vis.configure_traits(view=myview,context={'large_cm_scene':large_cm_scene.trait_view(),'vis':vis})



try:
    cm=worm['CM']
    vo=worm['Vol']
    me=worm['Med']
except:
    pass




opt_Vol_dir=Vol_dir+'_opt'


##myview=View('large_cm_scene')
#myview=View(
#    HGroup(
##        Item('large_cm_scene.scene',editor=SceneEditor(scene_class=MayaviScene),height=250, width=300, show_label=False),
#        'large_cm_scene',
#        'vis.x',
#    )
#)
#
#vis=testVis()
##vis.scene_group=HGroup(
##    Item('large_cm_scene.scene',editor=SceneEditor(scene_class=MayaviScene),height=250, width=300, show_label=False),
##    Item('small_cm_scene.scene',editor=SceneEditor(scene_class=MayaviScene),height=250, width=300, show_label=False),
##    )
