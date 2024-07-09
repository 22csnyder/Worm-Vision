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
from traitsui.api import RangeEditor



#import parent
from BaseClasses import tWorm
from traitsui.api import View,VGroup,HGroup,Include,TextEditor,Item,spring
from mayavi.core.ui.mayavi_scene import MayaviScene
from tvtk.pyface.scene_editor import SceneEditor
#from tvtk.pyface.list_editor import ListEditor
#from tvtk.pyface.list_editor import ListEditor
from traitsui.editors import ListEditor


def array_is_odd(arr):
    return (np.array(arr)%2==1).all()

#        self.floor_x_pos=Int(0-self.window_shape[0]//2-1)
#        self.floor_y_pos=0-self.window_shape[1]//2-1
#        self.floor_z_pos=0-self.window_shape[2]//2-1

def elementwise_max(arr1,arr2):
    return np.array([max(x1,x2) for x1,x2 in zip(arr1,arr2)])

def elementwise_min(arr1,arr2):
    return np.array([min(x1,x2) for x1,x2 in zip(arr1,arr2)])

#Try this later##


class MyRange(Range):
    def __init__(self,*args,**kwargs):
        Range.__init__(self,*args,**kwargs)
        self.enter_set=True
        self.auto_set=False
#        self.editor=RangeEditor(mode='xslider')
        self.mode='slider'

class RangePoint(HasTraits):
    z=MyRange(low='floor_z',high='ceil_z')
    y=MyRange(low='floor_y',high='ceil_y')
    x=MyRange(low='floor_x',high='ceil_x')
    floor_z=Int(0)
    floor_y=Int(0)
    floor_x=Int(0)
    ceil_z=Int(50)
    ceil_y=Int(50)
    ceil_x=Int(50)
    
    allowed_negative=Bool(False)

    def __init__(self,*args):
        HasTraits.__init__(self)
        
        self.set_value(*args)
    
    def set_value(self,*args):
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
    
    def upr_bdd_is_atleast(self,arr):
        cz,cy,cx=np.array(arr).astype(np.int)
        self.ceil_z=max(self.ceil_z,cz)
        self.ceil_y=max(self.ceil_y,cy)
        self.ceil_x=max(self.ceil_x,cx)
    
    def sync_traits(self,box_instance,trait_tuple):
        z,y,x,fz,fy,fx,cz,cy,cx=trait_tuple
        
        self.sync_trait('floor_z',box_instance,fz,mutual=True)
        self.sync_trait('floor_y',box_instance,fy,mutual=True)
        self.sync_trait('floor_x',box_instance,fx,mutual=True)
        
        self.sync_trait('ceil_z',box_instance,cz,mutual=True)
        self.sync_trait('ceil_y',box_instance,cy,mutual=True)
        self.sync_trait('ceil_x',box_instance,cx,mutual=True)

        self.sync_trait('z',box_instance,z,mutual=True)
        self.sync_trait('y',box_instance,y,mutual=True)
        self.sync_trait('x',box_instance,x,mutual=True)        

    def __setattr__(self,name,value):#if we would violate the range, just ignore that part

        if name in 'zyx':
            value=int(value)
            
            
        if name == 'z':
            self.ceil_z=max(self.ceil_z,value)
        if name == 'y':
            self.ceil_y=max(self.ceil_y,value)
        if name == 'x':
            self.ceil_x=max(self.ceil_x,value)
            
        if self.allowed_negative:
            if name == 'z':
                self.floor_z=min(self.floor_z,value)
            if name == 'y':
                self.floor_y=min(self.floor_y,value)
            if name == 'x':
                self.floor_x=min(self.floor_x,value)            
            

            
        super(HasTraits,self).__setattr__(name,value)

    def __repr__(self):
        return self.__array__().__repr__()
                
    def __array__(self):
        return np.array([self.z,self.y,self.x])

    def __getitem__(self,*args,**kwargs):
        return self.__array__().__getitem__(*args,**kwargs)

    def __add__(self,other):
        if hasattr(other,'z'):
            return np.array([self.z+other.z,
                             self.y+other.y,
                             self.x+other.x])
        elif isinstance(other,(int,long,float)):
            return np.array([self.z+other,self.y+other,self.x+other])
        else:
            return np.array([self.z+other[0],
                 self.y+other[1],
                 self.x+other[2]])
            

    def __sub__(self,other):
        if hasattr(other,'z'):
            return np.array([self.z-other.z,
                             self.y-other.y,
                             self.x-other.x])      
        elif isinstance(other,(int,long,float)):
            return np.array([self.z-other,self.y-other,self.x-other])
        else:
            return np.array([self.z-other[0],
                 self.y-other[1],
                 self.x-other[2]])

    view=View(VGroup(
                Item('x',label='Z'),
                Item('y',label='Y'),
                Item('z',label='X'),
            )
        )

#    view=View(VGroup(
#            'x','y','z',
#            )
#        )

#____________________________________________________________________________________________

class RawCoordinates(HasTraits):
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

    z_pos=MyRange(low='floor_z_pos',high='ceil_z_pos')
    y_pos=MyRange(low='floor_y_pos',high='ceil_y_pos')
    x_pos=MyRange(low='floor_x_pos',high='ceil_x_pos')
    floor_z_pos=Int(0)
    floor_y_pos=Int(0)
    floor_x_pos=Int(0)
    ceil_z_pos=Int(1)
    ceil_y_pos=Int(1)
    ceil_x_pos=Int(1)
    
    z_len=MyRange(low='floor_z_len',high='ceil_z_len')    
    y_len=MyRange(low='floor_y_len',high='ceil_y_len')
    x_len=MyRange(low='floor_x_len',high='ceil_x_len')
    floor_z_len=Int(1)
    floor_y_len=Int(1)
    floor_x_len=Int(1)
    ceil_z_len=Int(1)
    ceil_y_len=Int(1)
    ceil_x_len=Int(1)
    
    def __init__(self):
        HasTraits.__init__(self)

        self._tuple_center=['z_center','y_center','x_center',
                              'floor_z_center','floor_y_center','floor_x_center',
                              'ceil_z_center','ceil_y_center','ceil_x_center']
                              
        self._tuple_radius=['z_radius','y_radius','x_radius',
                              'floor_z_radius','floor_y_radius','floor_x_radius',
                              'ceil_z_radius','ceil_y_radius','ceil_x_radius']
                              
        self._tuple_position=['z_pos','y_pos','x_pos',
                              'floor_z_pos','floor_y_pos','floor_x_pos',
                              'ceil_z_pos','ceil_y_pos','ceil_x_pos']
                              
        self._tuple_length=['z_len','y_len','x_len',
                              'floor_z_len','floor_y_len','floor_x_len',
                              'ceil_z_len','ceil_y_len','ceil_x_len']
                              
#____________________________________________________________________________________________

class BoxCoords(RawCoordinates):
    dirty=Bool(False)#Consider making this an Event() which is write only
    flush_changes=Bool(False)

 
    image_shape=Array()
    window_shape=Array()
    
#    center=Instance(RangePoint,args=(20,40,01))
    
    
    def __init__(self,window_shape,image_shape,must_have_odd_shape=True):
        RawCoordinates.__init__(self)
        
        self.must_have_odd_shape=must_have_odd_shape
        if (not array_is_odd(window_shape)) and self.must_have_odd_shape:
            raise ValueError('window_shape must have all odd dimensions')
        
        self.image_shape=np.array(image_shape)
        self.window_shape=np.array(window_shape)
        
        
        self.window_radius=self.window_shape//2
        default_point=self.window_radius

#        self.center=RangePoint(default_point)
#        self.radius=RangePoint()
#        self.pos=RangePoint(0,0,0)
#        self.length=RangePoint(self.window_shape)

        self.add_trait('center',RangePoint(default_point))
        self.add_trait('radius',RangePoint(self.window_radius))
        self.add_trait('pos',RangePoint(0,0,0))
        self.add_trait('length',RangePoint(self.window_shape))

        self.pos.allowed_negative=True

        self.center.sync_traits(self,self._tuple_center)
        self.radius.sync_traits(self,self._tuple_radius)
        self.pos.sync_traits(self,self._tuple_position)
        self.length.sync_traits(self,self._tuple_length)



        self.ceil_z_pos,self.ceil_y_pos,self.ceil_x_pos=self.image_shape
        self.floor_z_pos,self.floor_y_pos,self.floor_x_pos= - self.window_radius

    
#        self.intended_radius=self.radius.__array__()


#        view_center=VGroup('x_center','y_center','z_center')
#        self.add_trait('view_center',view_center)
        
#        test_group=VGroup('floor_x_pos','floor_y_center')
#        self.add_trait('test_group',test_group)        
        
#        self.unit_cube=np.mgrid[0:2,0:2,0:2]        
#        self.unit_cube_vertices=self.unit_cube.reshape(-1,8).transpose()
#        self.vertices=np.copy(self.unit_cube_vertices)
#        self.sgrid=tvtk.StructuredGrid(dimensions=(2,2,2))
#        self.sgrid.point_data.scalars=np.ones(self.vertices.shape)
#        self.sgrid.point_data.scalars.name='cube_mesh_scalars'        
#        self.update_vertices()
#        self.grid_src=VTKDataSource(data=self.sgrid)
#    def update_vertices(self):
#        self.vertices=np.copy(self.unit_cube_vertices)
#        self.lengths=[self.x_len,self.y_len,self.z_len]
#        self.position=[self.x_pos,self.y_pos,self.z_pos]
#
#        for d in range(3):
#            self.vertices[:,d]*=self.lengths[d]
#            self.vertices[:,d]+=self.position[d]
#            
#            if hasattr(self,'window_shape'):
#                loc=self.vertices[:,d]>self.image_shape[d]
#                self.vertices[loc,d]=self.image_shape[d]
#            
#        self.sgrid.points=self.spacing*self.vertices




    
    @on_trait_change('image_shape')
    def set_center_range(self):
        self.ceil_z_center,self.ceil_y_center,self.ceil_x_center=self.image_shape
    
#    @on_trait_change('center')
    @on_trait_change('center.[z,y,x]')
    def on_center_change(self):
#        print 'set position fire'
#        print '    center is ',self.center.__array__()
#        print '    pos is ',self.pos.__array__()
#        print '    radius is ',self.radius.__array__()
        
        
        r_ceil=elementwise_min(self.window_radius+self.image_shape-self.center   ,   self.center+self.window_radius) 
        self.radius.set_value(elementwise_min(r_ceil,self.radius.__array__()))
#        self.radius.set_value(elementwise_min(r_ceil,self.window_radius))#re expand to window_radius if possible
        self.ceil_z_radius,self.ceil_y_radius,self.ceil_x_radius=r_ceil

        self.pos.set_value(self.center-self.radius)

    @on_trait_change('radius.[z,y,x]')
    def on_radius_change(self):
        self.floor_z_pos,self.floor_y_pos,self.floor_x_pos=-1*self.radius.__array__()
        self.length.set_value(2*self.radius.__array__()+1)

        self.pos.set_value(self.center-self.radius)

#    def _x_center_changed(self):
#        print 'x_center changed'    
    
    def move_to(self,*args):
        self.center.set_value(*args)
    
    def in_range(self,point):
        point=np.array(point)
        left_bdry_cond= (point>self.pos.__array__()).all()
        right_bdry_cond= (point<self.pos.__array__()+self.length.__array__()).all()
        if left_bdry_cond and right_bdry_cond:
            return True
        else:
            return False
            
            
        
    
#    def set_radius_range(self):#I'm allowing the regions to go past the image by "self.window_radius" amount
#        
#        ceil=elementwise_min(self.window_radius+self.image_shape-self.center   ,   self.center+self.window_radius)
#        
##        self.ceil_z_radius=min(self.window_radius[0]+self.image_shape[0]-self.z_center)
    
    
    
#    def check_bounds(self):
#        if self.x_pos+self.x_len > self.image_shape[2]+self.window_radius[2]:
#            self.x_pos=self.image_shape[2]-self.x_len
#        if self.y_pos+self.y_len > self.image_shape[1]:
#            self.y_pos=self.image_shape[1]-self.y_len
#        if self.z_pos+self.z_len > self.image_shape[0]:
#            self.z_pos=self.image_shape[0]-self.z_len
#            
#
#        if self.x_pos<0:self.x_pos=0
#        if self.y_pos<0:self.y_pos=0
#        if self.z_pos<0:self.z_pos=0


#    test_group=Instance(VGroup,value=('floor_x_pos','floor_y_center'))
    test_group=VGroup('floor_x_pos','floor_y_center')


    view_center=VGroup('x_center','y_center','z_center')
    view_radius=VGroup('x_radius','y_radius','z_radius')
    view_position=VGroup('x_pos','y_pos','z_pos')
    view_length=VGroup('x_len','y_len','z_len')
    
    
#    coordinate_control=VGroup(
#        HGroup(Item('object.center',style='custom'),
#        'object.radius'),
#        HGroup('object.pos',
#        'object.length'),    
#    )
#    coordinate_control=VGroup(
#        Item('object.center',style='custom',show_label=False),
#        'object.radius',
#        'object.pos',
#        'object.length',    
#    )
    coordinate_control=VGroup(
#        Item('object.center',style='custom',show_label=False),
        Item('object.center',style='custom',name='Center'),
        Item('object.radius',style='custom',name='Radius'),
        label='explorer',
    )

    view=View(Include('coordinate_control'))

#    traits_view=View(Include('coordinate_control'))
    
#    view=View('coordinate_control','floor_x_pos','floor_y_center')
        
        
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