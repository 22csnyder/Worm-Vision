# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 23:17:22 2015

@author: csnyder
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 09:38:41 2015

@author: csnyder
"""
import skimage.io as io
from mayavi import mlab
from scipy.ndimage import zoom

from traits.api import Array,NO_COMPARE,Enum

import numpy as np
from traits.api import HasTraits,on_trait_change,Instance,Button,Float,DelegatesTo,Str,Range,Int,Bool
from traitsui.api import View,VGroup,HGroup,Include,TextEditor,Item,spring
from traitsui.api import RangeEditor

from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from tvtk.pyface.scene_editor import SceneEditor

import time
from mayavi.core.api import Engine

from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.modules.structured_grid_outline import StructuredGridOutline
from mayavi.sources.api import ArraySource
from mayavi.modules.volume import Volume

from save_patches import parse_line,get_number
from multi_obj_polar_nucleus3D import PolarCache
from WormBox.BaseClasses import Environment#,Nucleus
from WormBox.WormSources2 import MultiVolume,CubeMesh,SphereMesh
import collections
import dill
from mayavi.modules.volume import Volume
from traits.has_dynamic_views import HasDynamicViews, DynamicView,DynamicViewSubElement

from apptools.scripting.api import Recorder,recordable,set_recorder



#debug_lut_array=[]





        
        
    #use tvtk.to_tvtk(vtk_obj) to get the tvtk version of this object
    
def ButtonEvent(obj,event):
    if event== "RightButtonPressEvent":
        print 'right button was pressed'
	        

class Vis(HasTraits):
    scene=Instance(MlabSceneModel,())

    def cbk(self,vtk_obj,event):
        hotkeys=dict(toggle_mesh='t')
        print vtk_obj.GetClassName(),event,vtk_obj.GetKeyCode()
        if vtk_obj.GetKeyCode()==hotkeys(['toggle_mesh']):       
            self._toggle_mesh_visibility_fired()
            
    def __init__(self,patch):
        HasTraits.__init__(self)
        

        self.Patch=patch
        
        self.array_src=ArraySource()        
        self.array_src.scalar_data=self.Patch
        self.scene.mayavi_scene.add_child(self.array_src)
        self.vol_module=Volume()
        self.volume=self.array_src.add_module(self.vol_module)
        
        
        
    @on_trait_change('scene.activated')
    def scene_activated(self):
        print 'scene is activated'
        iren=self.scene.scene.interactor
        iren.add_observer("RightButtonPressEvent",ButtonEvent)
        iren.add_observer("KeyPressEvent",self.cbk)
        print 'changed style'


    traits_view = View( Item('scene',editor=SceneEditor(scene_class=MayaviScene),height=250, width=300, show_label=False))

        

if __name__=='__main__':


    path='/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/patches/patch94.tif'

    #I=io.imread(patchpath)
    I0=io.imread(path)
    I=I0.astype(np.float)
    I*=(255/I.max())
    
    
    print 'finished reading data'
    
    vis=Vis(I)

    vis.configure_traits()
    
    

    
    #def picker_callback(picker):
    #    """ Picker callback: this get called when on pick events.
    #    """
    #    if picker.actor in red_glyphs.actor.actors:
    #        # Find which data point corresponds to the point picked:
    #        # we have to account for the fact that each data point is
    #        # represented by a glyph with several points
    #        point_id = picker.point_id/glyph_points.shape[0]
    #        # If the no points have been selected, we have '-1'
    #        if point_id != -1:
    #            # Retrieve the coordinnates coorresponding to that data
    #            # point
    #            x, y, z = x1[point_id], y1[point_id], z1[point_id]
    #            # Move the outline to the data point.
    #            outline.bounds = (x-0.1, x+0.1,
    #                              y-0.1, y+0.1,
    #                              z-0.1, z+0.1)
    
    
