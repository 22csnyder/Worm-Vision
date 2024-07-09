# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:59:47 2015

@author: csnyder
"""

import numpy as np
from mayavi import mlab
from traits.api import HasTraits, Range, Instance, \
                    on_trait_change
from traitsui.api import View, Item, HGroup,VGroup
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import \
                    MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from traits.api import Button

from traits.api import Float

from skimage import io

from tvtk.pyface.api import Scene

class Visualization(HasTraits):
    
    button=Button
    scene      = Instance(MlabSceneModel,())    
    integral=Float(1.2)
    
    
    def __init__(self,I):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        self.I=I
        self.sIzi,self.sIxi,self.sIyi=np.indices(self.I.shape)
        s=self.scene.mlab.pipeline.scalar_field(self.sIzi,self.sIxi,self.sIyi,self.I)
        self.volume=self.scene.mlab.pipeline.volume(s)

#        self.volume=mlab.pipeline.volume(mlab.pipeline.scalar_field(self.sIzi,self.sIxi,self.sIyi,self.I))
        self.volume.volume_property.scalar_opacity_unit_distance = 2.0

#Interesting: makes a cross section with crossshaped 'marker' at point
        self.cursor=mlab.pipeline.user_defined(s,filter='ImageCursor3D')
        self.cursor.filter.cursor_position = np.array([20, 20, 20])
        self.cursor.filter.cursor_value = 0
        self.cursor.filter.cursor_radius = 10
#        ipw = self.scene.mlab.pipeline.image_plane_widget(self.cursor, plane_orientation='x_axes',slice_index=20)
        
#    def _integral_changed()    
        
#        fig.scene.picker.pointpicker.add_observer('EndPickEvent', picker_callback)

    def picker_callback(picker_obj,event):
        print 'point picked!'
        
    def _button_fired():
        print 'button fired'
        
    view = View(Item('scene', 
            editor=SceneEditor(scene_class=MayaviScene),
#            editor=SceneEditor(scene_class=Scene),
                height=250, width=300, show_label=False),
            VGroup(
                    'integral','button'
                ),
            )

    picker=scene.mayavi_scene.on_mouse_pick(picker_callback)    


if __name__=='__main__':
    
    I=io.imread('/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/patch1.tif')
    
    
    vis = Visualization(I)
    vis.configure_traits()

