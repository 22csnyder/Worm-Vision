# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 08:59:56 2015

@author: csnyder
"""
import skimage.io as io
import numpy as np
from mayavi import mlab

from tvtk.api import tvtk
from traits.api import Range,Int,HasTraits,on_trait_change,Bool,Instance,List,Array
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
from mayavi.core.ui.mayavi_scene import MayaviScene
from tvtk.pyface.scene_editor import SceneEditor
#from tvtk.pyface.list_editor import ListEditor
#from tvtk.pyface.list_editor import ListEditor
from traitsui.editors import ListEditor


def array_is_odd(arr):
    return (arr%2==1).all()

#        self.floor_x_pos=Int(0-self.window_shape[0]//2-1)
#        self.floor_y_pos=0-self.window_shape[1]//2-1
#        self.floor_z_pos=0-self.window_shape[2]//2-1


#Try this later##


class MyRange(Range):
    def __init__(self,*args,**kwargs):
        Range.__init__(self,*args,**kwargs)
        self.enter_set=True
        self.auto_set=False
        self.mode='slider'

class RangePoint(HasTraits):
    ceil_z=Int(2)
    ceil_y=Int(222)
    ceil_x=Int(222)
    floor_z=Int(1)
    floor_y=Int(1)
    floor_x=Int(1)
    
    z=MyRange(low='floor_z',high='ceil_z')
    y=MyRange(low='floor_y',high='ceil_y')
    x=MyRange(low='floor_x',high='ceil_x')

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
    
    def sync_traits(self,box_instance,z,y,x,fz,fy,fx,cz,cy,cx):
#    def sync_traits(self,box_instance,z,y,x):
        self.sync_trait('z',box_instance,z,mutual=True)
        self.sync_trait('y',box_instance,y,mutual=True)
        self.sync_trait('x',box_instance,x,mutual=True)

    def __setattr__(self,name,value):#if we would violate the range, just ignore that part
        if name == 'z':
            self.ceil_z=max(self.ceil_z,value)
        if name == 'y':
            self.ceil_y=max(self.ceil_y,value)
        if name == 'x':
            self.ceil_x=max(self.ceil_x,value)
        super(HasTraits,self).__setattr__(name,value)
                


#____________________________________________________________________________________________

class RawCoordinates(HasTraits):
    z_len=Range(low='floor_z_len',high='ceil_z_len')    
    y_len=Range(low='floor_y_len',high='ceil_y_len')
    x_len=MyRange(low='floor_x_len',high='ceil_x_len')
    floor_z_len=Int(1)
    floor_y_len=Int(1)
    floor_x_len=Int(1)
    ceil_z_len=Int(1)
    ceil_y_len=Int(1)
    ceil_x_len=Int(1)
    
    z_pos=MyRange(low='floor_z_pos',high='ceil_z_pos')
    y_pos=MyRange(low='floor_y_pos',high='ceil_y_pos')
    x_pos=MyRange(low='floor_x_pos',high='ceil_x_pos')
    floor_z_pos=Int(0)
    floor_y_pos=Int(0)
    floor_x_pos=Int(0)
    ceil_z_pos=Int(1)
    ceil_y_pos=Int(1)
    ceil_x_pos=Int(1)


    z_center=MyRange(low='floor_z_center',high='ceil_z_center')
    y_center=MyRange(low='floor_y_center',high='ceil_y_center')
    x_center=MyRange(low='floor_x_center',high='ceil_x_center')
    floor_z_center=Int(0)
    floor_y_center=Int(0)
    floor_x_center=Int(0)
    ceil_z_center=Int(1)
    ceil_y_center=Int(1)
    ceil_x_center=Int(1)

    z_radius=MyRange(low='floor_z_radius',high='ceil_z_radius')
    y_radius=MyRange(low='floor_y_radius',high='ceil_y_radius')
    x_radius=MyRange(low='floor_x_radius',high='ceil_x_radius')
    floor_z_radius=Int(0)      
    floor_y_radius=Int(0)
    floor_x_radius=Int(0)
    ceil_z_radius=Int(1)
    ceil_y_radius=Int(1)
    ceil_x_radius=Int(1)

class BoxCoords(HasTraits):
    dirty=Bool(False)#Consider making this an Event() which is write only
    flush_changes=Bool(False)

 
        
    image_shape=Array()
    window_shape=Array()
    
    def __init__(self,window_shape,image_shape):
        
        if not array_is_odd(window_shape):
            raise ValueError('window_shape must have all odd dimensions')
        
        self.window_shape=window_shape
        self.image_shape=image_shape        
        
        self.add_trait('Position',List([self.z_pos,self.y_pos,self.x_pos]))
        self.add_trait('length',List([self.z_len,self.y_len,self.x_len]))
        
        self.window_shape=window_shape
        self.window_radius=self.window_shape//2

        self.ceil_z_pos,self.ceil_y_pos,self.ceil_x_pos=self.image_shape


        self.pos=RangePoint(22,23,24)        
        self.pos.sync_traits(self,'z_pos','y_pos','x_pos')



#        self.pos.sync_traits(self,self.z_pos,self.y_pos,self.x_pos,fz,fy,fx,z,y,x):
#        self.sync_trait('z',box_instance,'z',mutual=True)
#        self.sync_trait('y',box_instance,'y',mutual=True)
#        self.sync_trait('x',box_instance,'x',mutual=True)
#        self.ceil_z_len,self.ceil_y_len,self.ceil_x_len=self.image_shape

        
#        self.ceil_z_len,self.ceil_y_len,self.ceil_x_len=self.window_shape####init method
    

    
    on_trait_change('image_shape')
    def set_center_range(self):
        self.ceil_z_pos,self.ceil_y_pos,self.ceil_x_pos=self.image_shape
    
    on_trait_change('z_center','y_center','x_center')
    def set_radius_range(self):
        
        self.ceil_z_radius=min(self.window_radius[0]+self.window_shape[0]-self.z_center)
    
    
    
    def check_bounds(self):
        if self.x_pos+self.x_len > self.image_shape[2]+self.window_radius[2]:
            self.x_pos=self.image_shape[2]-self.x_len
        if self.y_pos+self.y_len > self.image_shape[1]:
            self.y_pos=self.image_shape[1]-self.y_len
        if self.z_pos+self.z_len > self.image_shape[0]:
            self.z_pos=self.image_shape[0]-self.z_len
            

        if self.x_pos<0:self.x_pos=0
        if self.y_pos<0:self.y_pos=0
        if self.z_pos<0:self.z_pos=0





    view_position=VGroup('x_pos','y_pos','z_pos')
    view_length=VGroup('x_len','y_len','z_len')
    
    

    coordinate_control=HGroup(
        view_position,
        view_length,

#        'x_pos',
#        'pos',
#        'location',
    
    
    )

    traits_view=View(Include('coordinate_control'))
    
    
        
        
#        if self.x_pos+self.x_len > self.window_shape[0]:
#            self.x_pos=self.window_shape[0]-self.x_len
#        if self.y_pos+self.y_len > self.window_shape[1]:
#            self.y_pos=self.window_shape[1]-self.y_len
#        if self.z_pos+self.z_len > self.window_shape[2]:
#            self.z_pos=self.window_shape[2]-self.z_len
#    def check_bounds(self):
##        if self.x_pos+self.x_len > self.window_shape[0]:
##            self.x_pos=self.window_shape[0]-self.x_len
##        if self.y_pos+self.y_len > self.window_shape[1]:
##            self.y_pos=self.window_shape[1]-self.y_len
##        if self.z_pos+self.z_len > self.window_shape[2]:
##            self.z_pos=self.window_shape[2]-self.z_len
##        if self.x_pos<0:self.x_pos=0
##        if self.y_pos<0:self.y_pos=0
##        if self.z_pos<0:self.z_pos=0
#        if self.x_pos+self.x_len > self.image_shape[0]:
#            self.x_pos=self.image_shape[0]-self.x_len
#        if self.y_pos+self.y_len > self.image_shape[1]:
#            self.y_pos=self.image_shape[1]-self.y_len
#        if self.z_pos+self.z_len > self.image_shape[2]:
#            self.z_pos=self.image_shape[2]-self.z_len
#        if self.x_pos<0:self.x_pos=0
#        if self.y_pos<0:self.y_pos=0
#        if self.z_pos<0:self.z_pos=0
#    @on_trait_change('dirty')##This is triggered after all coordinate changes are done.
#    def update(self):
#        if self.dirty:
##            print 'CubeMesh.update()'
#            self.check_bounds()
#            self.update_vertices()
#            self.update_data()
##            print '   -resultant I.mean=',np.mean(self.I)
#            self.dirty=False
#            #calculate bounds()--implement
#    def set_slider_limits(self):
#        self.ceil_x_len=self.image_shape[0]#It looks like "xyz" but it's really "zyx"
#        self.ceil_y_len=self.image_shape[1]
#        self.ceil_z_len=self.image_shape[2]
#        self.ceil_x_pos=self.image_shape[0]
#        self.ceil_y_pos=self.image_shape[1]
#        self.ceil_z_pos=self.image_shape[2]
#        self.floor_x_pos=0-self.window_shape[0]//2-1
#        self.floor_y_pos=0-self.window_shape[1]//2-1
#        self.floor_z_pos=0-self.window_shape[2]//2-1
#        #Make starting guess
#        self.x_len=self.window_shape[0]
#        self.y_len=self.window_shape[1]
#        self.z_len=self.window_shape[2]