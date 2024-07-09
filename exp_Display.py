# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 07:17:49 2015

@author: csnyder
"""

import skimage.io as io
import numpy as np
from mayavi import mlab

from tvtk.api import tvtk
from traits.api import Range,Int,HasTraits,on_trait_change,Bool,Instance,List
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.modules.structured_grid_outline import StructuredGridOutline
from mayavi.sources.api import ArraySource
from mayavi.modules.volume import Volume
from numpy.lib import stride_tricks
from mayavi import mlab
#from multi_obj_polar_nucleus3D import PolarMesh
from traitsui.api import RangeEditor

from traits.api import HasTraits,on_trait_change,Instance,Button,Float,DelegatesTo,Str,Range,Int,Bool
from mayavi.core.api import Engine
from mayavi.tools.mlab_scene_model import MlabSceneModel


#from Interactor import Interactor
from WormBox.BaseClasses import tWorm
from WormBox.VisClasses import VisWorm

from BoxCoords import BoxCoords

from traitsui.api import View,VGroup,HGroup,Include,TextEditor,Item,spring
from mayavi.core.ui.mayavi_scene import MayaviScene
from tvtk.pyface.scene_editor import SceneEditor
#from tvtk.pyface.list_editor import ListEditor
#from tvtk.pyface.list_editor import ListEditor
from traitsui.editors import ListEditor

from traits.api import HasTraits,Any


def elementwise_max(arr1,arr2):
    return np.array([max(x1,x2) for x1,x2 in zip(arr1,arr2)])

def elementwise_min(arr1,arr2):
    return np.array([min(x1,x2) for x1,x2 in zip(arr1,arr2)])


class Point(HasTraits):
    z=Int()
    y=Int()
    x=Int()
    def __init__(self,*args):
        HasTraits.__init__(self)
        if len(args)==1:
            self.z,self.y,self.x=args[0]
        elif len(args)==3:
            self.z,self.y,self.x=args
#            self.z=args[0]
#            self.y=args[1]
#            self.x=args[2]
        elif len(args)==0:
            self.z,self.y,self.x=0,0,0
        else:
            raise ValueError('need to pass tuple or 3 separate coordinates')

#from BoxCoords import elementwise_max

from SmartScene import SmartScene


#Maybe move to PyQt in the future
#Couldn't avoid inheriting VisWorm Directly :(
class Display(HasTraits):

    
#    max_time=Int(1)
#    min_time=Int(0)
#    time=Range('min_time','max_time',mode='slider')
#    
#    do_global_iteration=Button()
#    next_iteration=Button()#do iteration
#    reset_current_mesh=Button('retry')
#    
#    Average_Intensity=Float()
#    
#    radius_best_guess=Range(low=1.0,high=15.0,)        
#    emphasis_on_radius_guess=Range(low=0.0,high=60.0)
#    smoothness=Range(low=0.0,high=500.0)     
#    emphasis_on_marker_guess=Range(low=0.0,high=500.0)
#    step_size=Range(0.001,0.1,)    
#    
#    toggle_mesh_visibility=Button('on/off')
#    
#    neuron_name=Str('')
#    m=Int(-1,record=True)
##    n_ind=Int(-1)
#    
#    next_marker=Button()
#    previous_marker=Button()
#    save_session=Button() ##Button as of yet unimplemented
#    save_at_end=Bool(False)    
#    
#    down=Button('sample') 
#    up=Button('sample')
#
#    forward=Button('+')
#    backward=Button('-')
#
#    debug=Button('debug')    
#    
#    to_var=Button()
#    to_normal=Button()
#    
#    engine1 = Instance(Engine, args=())
#    engine2 = Instance(Engine, args=())
#    whole_worm      = Instance(MlabSceneModel,())
#    small_scene     = Instance(MlabSceneModel,())
#    
#    mesh_visibility=Bool(True)
#    
#    current_sphere_mesh= Instance(SphereMesh)
#    
##    integral=Float(1.2)
#
##    cube=Instance(CubeMesh(),())
#    
#    volume_control=Instance(Volume,())
#
#    cube=Instance(CubeMesh,())#Either seem to work
#
#    box_x_len=DelegatesTo(delegate='cube',prefix='x_len')
#    box_y_len=DelegatesTo(delegate='cube',prefix='y_len')
#    box_z_len=DelegatesTo(delegate='cube',prefix='z_len')
#    box_x_pos=DelegatesTo(delegate='cube',prefix='x_pos')
#    box_y_pos=DelegatesTo(delegate='cube',prefix='y_pos')
#    box_z_pos=DelegatesTo(delegate='cube',prefix='z_pos')
#    
#    floor_x_len=DelegatesTo('cube')
#    floor_y_len=DelegatesTo('cube')
#    floor_z_len=DelegatesTo('cube')
#    floor_x_pos=DelegatesTo('cube')
#    floor_y_pos=DelegatesTo('cube')
#    floor_z_pos=DelegatesTo('cube')
#
#    ceil_x_len=DelegatesTo('cube')
#    ceil_y_len=DelegatesTo('cube')
#    ceil_z_len=DelegatesTo('cube')
#    ceil_x_pos=DelegatesTo('cube')
#    ceil_y_pos=DelegatesTo('cube')
#    ceil_z_pos=DelegatesTo('cube')     
    
#    cube_display=VGroup('box_coords.x_len','box_coords.y_len','box_coords.z_len','box_coords.x_pos','box_coords.y_pos','box_coords.z_pos')
    
#    bad_marker=Button('mark as bad')
#    marker_status_options=['','marker is bad']
#    marker_status=Enum(marker_status_options)
#    
#    init_is_finished=Bool(False)
#    save_status_options=['session will not save at end','session will save at end']
#    save_status=Enum('session will not save at end',['session will not save at end','session will save at end'])
#    
#    show_right_panel=Button('color control');display_right_panel=Bool(False)
#    show_global_view=Button('global view');display_global_view=Bool(False)
#    has_time_component=Bool(True)
    
#    def _whole_worm_default(self):
#        self.engine1.start()
#        return MlabSceneModel(engine=self.engine1)
#
#    def _small_scene_default(self):
#        self.engine2.start()
#        return MlabSceneModel(engine=self.engine2)
    engine1 = Instance(Engine, args=())
    engine2 = Instance(Engine, args=())
    
    small_scene=Instance(SmartScene,())
    large_scene=Instance(SmartScene,())
#_____________
    worm=Instance(VisWorm,())
    
#    cc=Any()    
    
#    center=Instance(Point())

    def _large_scene_default(self):
        self.engine1.start()
        return SmartScene(engine=self.engine1)

    def _small_scene_default(self):
        self.engine2.start()
        return SmartScene(engine=self.engine2)

    def __init__(self,window_shape=None,spacing=None,smear=None,*worm_args):

#        VisWorm.__init__(self,window_shape,spacing,smear,*worm_args)    
        self.worm=VisWorm(window_shape,spacing,smear,*worm_args)
        
#        self.cc=Any()
##        self.worm.sync_trait('view_center',self,'cc',mutual=True)
#        self.sync_trait('cc',self.worm,'view_center',mutual=True)
        
        
        
        self.worm.display_on_large_scene(self.large_scene)
        
#        self.worm.display_on
        
        
#        coordinate_control=HGroup(
#            
#            view_center,
#            view_radius,        
#            view_position,
#            view_length,
#        )        
        
        if window_shape is None:
            self.window_shape=np.array(self.worm.X.shape)
        else:
            self.window_shape=np.array(window_shape)

        
        
                

    

        

        
        
    traits_view = View( 
        HGroup(
#            Include('coordinate_control'),
            VGroup(
                Item('large_scene',editor=SceneEditor(scene_class=MayaviScene),height=250, width=300, show_label=False),
#                Item('object.worm.coordinate_control'),
#                Item('object.worm.ceil_x_pos'),
                Item('object.worm',label='FOV'),

#                Item('object.worm.test_group'),
#                Instance(Item('object.worm.test_group')),
#                Item(Instance('object.worm.test_group')),
#                Item(Instance(VGroup, value=('object.worm.test_group'))),
#                Instance(Item(VGroup, value=(Item('object.worm.test_group')))),
#                Instance('object.worm.test_group'),
#                'cc'
#                Item('debug_string',show_label=False,editor=TextEditor(auto_set=False,enter_set=True,evaluate=str)),
                
            ),
        )
    )        



if __name__=='__main__':

    path='/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/patches/patch94.tif'
    
    #I=io.imread(patchpath)
    I0=io.imread(path)
    I=I0.astype(np.float)
    I*=(255/I.max())    
    I=I[:,10:65,:]
  
#    im = mlab.imshow(I[16])
#    cursor = mlab.points3d(0, 0, 0, mode='2dthick_cross',
#                           color=(0, 0, 0),
#                            scale_factor=10)        
#    from LocalWindow import LocalWindow

#    vis=Display(I.shape,I)
    vis=Display([11,11,11],None,None,I)
    
    
#    vis.center.set_value([2,54,12])
    
    
#    vis.center.set_value([22,32,311])

    
#    vis.get_sphere_patch()
#    d=vis
#    worm=vis.worm
#    c=vis.worm.coordinate_control
    
#    vis.configure_traits(view=View(VGroup(
#    Item('d.scene',editor=SceneEditor(scene_class=MayaviScene),height=250, width=300, show_label=False),
#    'worm.x_pos')), context={'d':vis,'worm':vis.worm})
    
    
#    vis.configure_traits(view=View(VGroup(
#    Item('scene',editor=SceneEditor(scene_class=MayaviScene),height=250, width=300, show_label=False),
#    'worm.coordiante_control')))
    
    vis.configure_traits()
    
    