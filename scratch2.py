# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:42:43 2015

@author: csnyder
"""


import skimage.io as io
import numpy as np
from mayavi import mlab

from tvtk.api import tvtk
from traits.api import Range,Int,HasTraits,HasPrivateTraits,on_trait_change,Bool,Instance,List,Array
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.modules.structured_grid_outline import StructuredGridOutline
from mayavi.sources.api import ArraySource
from mayavi.modules.volume import Volume
from numpy.lib import stride_tricks
from mayavi import mlab
from multi_obj_polar_nucleus3D import PolarMesh
from traitsui.api import RangeEditor



#import parent
from Interactor import Interactor
from WormBox.BaseClasses import tWorm
from traitsui.api import View,VGroup,HGroup,Include,TextEditor,Item,spring

from tvtk.pyface.scene_editor import SceneEditor
#from tvtk.pyface.list_editor import ListEditor
#from tvtk.pyface.list_editor import ListEditor
from traitsui.editors import ListEditor

from mayavi.tools.sources import MGlyphSource
from mayavi.modules.glyph import Glyph
from mayavi.modules.vectors import Vectors
from mayavi.sources.builtin_surface import BuiltinSurface
from mayavi.modules.surface import Surface

from WormScene import WormScene
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
#from mayavi.core.api import Engine
from mayavi.core.engine import Engine
from mayavi.core.scene import Scene


###ERROR: The SceneModel may only have one active editor

path='/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/patches/patch94.tif'

#I=io.imread(patchpath)
I0=io.imread(path)
I=I0.astype(np.float)
I*=(255/I.max())

I=I[:,10:65,:]

#engine=Engine()
#engine.start()
#
##scene=WormScene()
##scene=MlabSceneModel()
#scene=MayaviScene(engine)


#scene=Scene()
#a=ArraySource()
#a.scalar_data=I
##scene.mayavi_scene.add_child(a)
##scene.add_actor(a)
#a.add_module(Volume())
##mlab.title('title',figure=scene)
##mlab.text(.05,.05,'sample')
#scene.start()
#scene.configure_traits()


s=mlab.pipeline.scalar_field(I)
v=mlab.pipeline.volume(s)
#mlab.title('title')
#mlab.text(0.02,0.95,'pretty long text')#top left

#from mayavi.tools.sources import MGlyphSource


#g=MGlyphSource()

#a=ArraySource()
#
#p=[[4,3,2],[1,1,1],[2,2,2]]
##p=np.array([[3,3,3],[1,2,3],[1,1,1],[1,1,1]])
##p=np.array([[3,3,3,1,2,3],[1,1,1,1,1,1]])
#
##p=[[4,3,2]]
#a.scalar_data=p
#a.scalar_name='snd'







#class Neuron(BuiltinSurface):
#    surface=Instance(Surface,())
#    def __init__(self,*args,**kwargs):
#        BuiltinSurface.__init__(self,*args,**kwargs)
#        self.name='bis'
#        self.source='sphere'
#        self.data_source.center=[4,3,2]
#        self.surface=Surface()
##        self.surface.actor.property.color=(1.0,0.0,0.0)##This doesn't work if you do it here
#        self.add_module(self.surface)
#
#
##b= BuiltinSurface()
##b.name='bis'
##b.source='sphere'
##b.data_source.center=[4,3,2]
##s=Surface()
##b.add_module(s)
##s.actor.property.color=(1.0,0.0,0.0)
#
##scene.add_child(b)
#n=Neuron()
#scene.add_child(n)


#b.data_source.center












#pd=tvtk.PolyData()
#pd.set(points=p)

#scene.add_child(a)

#g=Vectors()
#a.add_module(g)



#g.glyph.color_mode='no_coloring'
#g.glyph.color_mode='color_by_scalar'
#g.glyph.glyph.scale_mode='data_scaling_off'
#g.glyph.glyph_source.glyph_source = g.glyph.glyph_source.glyph_dict['sphere_source']



#glyph = engine.scenes[0].children[0].children[0].children[0]
#glyph.actor.mapper.progress = 1.0
#glyph.actor.mapper.scalar_range = array([ 1.,  3.])
#glyph.actor.mapper.scalar_visibility = 0
#glyph.glyph.color_mode = 'no_coloring'
#glyph.glyph.glyph.range = array([ 1.,  3.])
#glyph.glyph.glyph.progress = 1.0
#glyph.glyph.glyph.scale_mode = 'data_scaling_off'
#glyph.glyph.glyph.range = array([ 0.,  1.])
#glyph.glyph.glyph.range = array([ 0.,  1.])
#glyph.glyph.scale_mode = 'data_scaling_off'
#glyph.glyph.glyph_source.glyph_source = glyph.glyph.glyph_source.glyph_dict['sphere_source']
#glyph.glyph.glyph.range = array([ 0.,  1.])
## glyph.glyph.glyph.source = <tvtk.tvtk_classes.poly_data.PolyData object at 0x2aaab24a9cb0>
## glyph.glyph.glyph_source.glyph_source = <tvtk.tvtk_classes.sphere_source.SphereSource object at 0x2aaab009cc50>
#glyph.glyph.glyph_source.glyph_source = glyph.glyph.glyph_source.glyph_dict['cylinder_source']
## glyph.glyph.glyph_source.glyph_source = <tvtk.tvtk_classes.cylinder_source.CylinderSource object at 0x2aaab009c8f0>
#glyph.glyph.glyph_source.glyph_source = glyph.glyph.glyph_source.glyph_dict['sphere_source']
## glyph.glyph.glyph_source.glyph_source = <tvtk.tvtk_classes.sphere_source.SphereSource object at 0x2aaab009cc50>
## glyph.glyph.mask_points.input = <tvtk.tvtk_classes.image_data.ImageData object at 0x2aaab0031050>
#glyph.glyph.mask_input_points = True
## glyph.glyph.glyph.input = <tvtk.tvtk_classes.image_data.ImageData object at 0x2aaab0031050>
#glyph.glyph.mask_input_points = False




#class member:
#    m='m'
#    def __init__(self):
#        pass
##class Child(HasTraits):
#class Child(HasPrivateTraits):
#    y_=Int(1)
#    M=Instance(member,())
#    view=Instance(View,())
##    x = 0
#    def __init__(self):
#        pass
#c=Child()
#
#c.y_1=1
#c.y_2=2
#c.y_3=3
#
#c.view=View(HGroup( 'y_1','y_2','y_3'))


##############################


#class VisWorm(tWorm,Box):
#    
#    downsample=Int(1)
#    
#
#    window_shape=Array()    
##    spacing=Array()#defined in tWorm
##    smear=Instance(Array,([1.0,1.0,1.0]))#registering as list bleh
#    t=Int(0)    
#    vol_module=Instance(Volume,())
#    
#    def __init__(self,Env,spacing=None):
#        self.nm_window_shape=nm_window_shape
#        
#        tWorm.__init__(self,Env)        
#        self.window_shape=np.round( np.array(self.nm_window_shape)/self.nm_voxel_shape )
#        self.image_shape=np.array(self.X.shape)
#        if not hasattr(self,'ini_file'):
#            if spacing is not None:           
#                self.spacing=spacing
#            else:
#                print 'warning, no way to determine pixel spacing...assuming uniform'
#        
##        if smear is not None:
##            self.smear=smear
##        else:
#
#        self.smear=np.array([1.0,1.0,1.0])#Fix later
#        
#        
#        self.window_shape=self.nm_window_shape/self.nm_voxel_shape
#        
#        
#    def assign_Scenes(self,Scenes):
#        
#        self.Scenes=Scenes
#         
#        self.Grid_Scenes=[]
#        if 'large scene' in self.Scenes.keys():
#            self.display_grid_on_scene(self.Scenes['large scene'])
#        
#        for scene in self.Scenes.values():
#            self.display_volume_on_scene(scene)
#            
#        
#    def read_markers(self,data):
#        self.apo=ApoReader(data)
#        self.apo.add_data_to_dict(self.segments)
#
#    def get_unscaled_padded_patch(self,point=None,radius_size=None):
#        radius_size=self.radius
#        if point==None:
#            point=self.center.__array__()
#        point=np.array(point)
#        zero=np.zeros_like(point)
#        upr=elementwise_min(self.image_shape,point+self.radius+1)
#        lwr=elementwise_max(zero,point-radius_size)
#        upad=elementwise_max(zero, -self.image_shape+(point+radius_size+1)  )
#        lpad=elementwise_max(zero,-point+radius_size)
#        self.padd=tuple([(l,u) for l,u in zip(lpad,upad)])
#        patch=np.array(self.X[lwr[0]:upr[0],lwr[1]:upr[1],lwr[2]:upr[2]])       
#        padded_patch=np.pad(patch,self.padd,mode='constant',constant_values=0)
#        return padded_patch
#        
##    def get_window_sized_unscaled_padded_patch(self,point=None):
##        if point==None:
##            point=self.center.__array__()
##        point=np.array(point)
##        zero=np.zeros_like(point)
##        upr=elementwise_min(self.image_shape,point+self.window_radius+1)
##        lwr=elementwise_max(zero,point-self.window_radius)
##        upad=elementwise_max(zero, -self.image_shape+(point+self.window_radius+1)  )
##        lpad=elementwise_max(zero,-point+self.window_radius)
##        self.padd=tuple([(l,u) for l,u in zip(lpad,upad)])
##        patch=np.array(self.X[lwr[0]:upr[0],lwr[1]:upr[1],lwr[2]:upr[2]])       
##        padded_patch=np.pad(patch,self.padd,mode='constant',constant_values=0)
##        return padded_patch
#    
#    def scale_data(self,arr):
#        darr=arr.astype(np.float)
#        M=np.max(darr)
#        if M==0:
#            dst=arr
#            scale=-1
#        else:
#            dst=255*darr/M
#            scale=M/255.0
#        return dst,scale
#
#
#    def display_volume_on_scene(self,scene,*args):
#        if scene.name == 'small scene':
#            self.display_on_small_scene(scene,*args)
#        elif scene.name == 'large scene':
#            downsample=scene.downsample
#            self.display_on_large_scene(scene,downsample,*args)
#            
#            
#    def display_on_small_scene(self,scene,t=0):#no option for downsample at the moment
#        _patch=self.get_unscaled_padded_patch()
#        patch,self.small_scene_scale=self.scale_data(_patch)
#
#        self.small_array_src=ArraySource(spacing=self.spacing)
#        self.small_array_src.scalar_data=patch
#        scene.mayavi_scene.add_child(self.small_array_src)
#        self.small_vol_module=Volume()
#        self.small_volume=self.small_array_src.add_module(self.small_vol_module)
##        self.small_volume=self.small_array_src.add_module(self.vol_module)
#        
#    def display_on_large_scene(self,scene,downsample=1,t=0):#entire view, possibly scaled down
#        print 'visclasses296 large_scene_display'
#        f=downsample#take every fth element #just in x,y planes
#        if self.X.ndim is 4:
#            dst=self.X[t]
#        else:
#            dst=self.X
#
#        patch,self.large_scene_scale=self.scale_data(np.array(dst[:,::f,::f]))#z,y,x
#        
#        self.large_array_src=ArraySource(spacing=self.spacing)
#        self.large_array_src.scalar_data=patch
#        scene.mayavi_scene.add_child(self.large_array_src)
#        self.large_vol_module=Volume()
#        self.large_volume=self.large_array_src.add_module(self.large_vol_module)        
#        
#        scene.mayavi_scene.add_child(self.large_array_src)
#        self.large_volume=self.large_array_src.add_module(self.large_vol_module)
##        self.large_volume=self.large_array_src.add_module(self.vol_module)
#
#
#    def display_grid_on_scene(self,scene):
#        self.Grid_Scenes.append(scene)
#        scene.mayavi_scene.add_child(self.grid_src)
#        structured_grid_outline=StructuredGridOutline()
#        self.grid_src.add_module(structured_grid_outline)
##############################

#c.y_8=8
#d.y_8#still 1

#Child.y_9=9
#d.y_9#now is 9






#from BoxCoords import RawCoordinates,array_is_odd
#
#class BoxCoords(RawCoordinates):
#    dirty=Bool(False)#Consider making this an Event() which is write only
#    flush_changes=Bool(False)
#
# 
#    image_shape=Array()
#    window_shape=Array()
#    
##    center=Instance(RangePoint,args=(20,40,01))
#    
#    
#    def __init__(self,window_shape,image_shape):
#        RawCoordinates.__init__(self)
#        
#        if not array_is_odd(window_shape):
#            raise ValueError('window_shape must have all odd dimensions')
#        
#        self.image_shape=np.array(image_shape)
#        self.window_shape=np.array(window_shape)
#        self.window_radius=self.window_shape//2
#        self.ceil_z_pos,self.ceil_y_pos,self.ceil_x_pos=self.image_shape
#
#        default_point=self.window_radius
#
##        self.center=RangePoint(default_point)
##        self.radius=RangePoint()
##        self.pos=RangePoint(0,0,0)
##        self.length=RangePoint(self.window_shape)
#
##        self.add_trait('center',RangePoint(default_point))
##        self.add_trait('radius',RangePoint(self.window_radius))
##        self.add_trait('pos',RangePoint(0,0,0))
##        self.add_trait('length',RangePoint(self.window_shape))
##
##        self.center.sync_traits(self,self._tuple_center)
##        self.radius.sync_traits(self,self._tuple_radius)
##        self.pos.sync_traits(self,self._tuple_position)
##        self.length.sync_traits(self,self._tuple_length)
#
#
#
#
#
#    
#    on_trait_change('image_shape')
#    def set_center_range(self):
#        print 'setting center range'
#        self.ceil_z_pos,self.ceil_y_pos,self.ceil_x_pos=self.image_shape
#    
##    on_trait_change('center')
##    on_trait_change('center.[z,y,x]')Doesn't work. I think maybe has to be List or Dicttrait
##    on_trait_change('floor_z_center')
#    on_trait_change(['x_center','self.y_center','z_center'])
#    def set_position(self):
#        print 'set position fired'
#        self.pos.set_value(self.center-self.radius)
#        
#
#    def _x_center_changed(self):
#        print 'x_center changed'    
#    
#    def move_to(self,*args):
#        self.center.set_value(*args)
#    
#
#
#B=BoxCoords([15,13,11],[50,60,70])








#        self._tuple_center=[self.z_center,self.y_center,self.x_center,
#                              self.floor_z_center,self.floor_y_center,self.floor_x_center,
#                              self.ceil_z_center,self.ceil_y_center,self.ceil_x_center]
#                              
#        self._tuple_radius=[self.z_radius,self.y_radius,self.x_radius,
#                              self.floor_z_radius,self.floor_y_radius,self.floor_x_radius,
#                              self.ceil_z_radius,self.ceil_y_radius,self.ceil_x_radius]
#                              
#        self._tuple_position=[self.z_pos,self.y_pos,self.x_pos,
#                              self.floor_z_pos,self.floor_y_pos,self.floor_x_pos,
#                              self.ceil_z_pos,self.ceil_y_pos,self.ceil_x_pos]
#                              
#        self._tuple_length=[self.z_len,self.y_len,self.x_len,
#                              self.floor_z_len,self.floor_y_len,self.floor_x_len,
#                              self.ceil_z_len,self.ceil_y_len,self.ceil_x_len]
#                              
#                              
#    x_len=Range('floor_x_len',high='ceil_x_len',value=1,enter_set=True,auto_set=False,mode='slider')
#    y_len=Range('floor_y_len',high='ceil_y_len',value=1,enter_set=True,auto_set=False,mode='slider')
#    z_len=Range('floor_z_len',high='ceil_z_len',value=1,enter_set=True,auto_set=False,mode='slider')    
#    floor_x_len=Int(1)
#    floor_y_len=Int(1)
#    floor_z_len=Int(1)
#    ceil_x_len=Int(1)
#    ceil_y_len=Int(1)
#    ceil_z_len=Int(1)
#    
##    x_pos=Range('floor_x_pos',high='ceil_x_pos',value=0,enter_set=True,auto_set=False,mode='slider')
##    y_pos=Range('floor_y_pos',high='ceil_y_pos',value=0,enter_set=True,auto_set=False,mode='slider')
##    z_pos=Range('floor_z_pos',high='ceil_z_pos',value=0,enter_set=True,auto_set=False,mode='slider')
#    x_pos=MyRange(low='floor_x_pos',high='ceil_x_pos')
#    y_pos=MyRange('floor_y_pos',high='ceil_y_pos')
#    z_pos=MyRange(low='floor_z_pos',high='ceil_z_pos')
#    ceil_x_pos=Int(222)
#    ceil_y_pos=Int(222)
#    ceil_z_pos=Int(222)
#    floor_x_pos=Int(0)
#    floor_y_pos=Int(0)
#    floor_z_pos=Int(0)
#
#    x_center=Range('floor_x_center',high='ceil_x_center',value=0,enter_set=True,auto_set=False,mode='slider')
#    y_center=Range('floor_y_center',high='ceil_y_center',value=0,enter_set=True,auto_set=False,mode='slider')
#    z_center=Range('floor_z_center',high='ceil_z_center',value=0,enter_set=True,auto_set=False,mode='slider')
#    ceil_x_center=Int(1)
#    ceil_y_center=Int(1)
#    ceil_z_center=Int(1)
#    floor_x_center=Int(0)
#    floor_y_center=Int(0)
#    floor_z_center=Int(0)
#
#    x_radius=Range('floor_x_radius',high='ceil_x_radius',value=0,enter_set=True,auto_set=False,mode='slider')
#    y_radius=Range('floor_y_radius',high='ceil_y_radius',value=0,enter_set=True,auto_set=False,mode='slider')
#    z_radius=Range('floor_z_radius',high='ceil_z_radius',value=0,enter_set=True,auto_set=False,mode='slider')
#    ceil_x_radius=Int(1)
#    ceil_y_radius=Int(1)
#    ceil_z_radius=Int(1)
#    floor_x_radius=Int(0)
#    floor_y_radius=Int(0)
#    floor_z_radius=Int(0)    