# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 08:42:26 2015

@author: csnyder
"""
import numpy as np
from tvtk.api import tvtk
from traits.api import Range,Int,HasTraits,on_trait_change,Bool,Instance
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.modules.structured_grid_outline import StructuredGridOutline
from mayavi.sources.api import ArraySource
from mayavi.modules.volume import Volume
from numpy.lib import stride_tricks
from mayavi import mlab
from multi_obj_polar_nucleus3D import PolarMesh
from traitsui.api import RangeEditor

class SphereMesh(PolarMesh,HasTraits):#Assumes all mesh patches to have the same shape
    def __init__(self,cache,patch,spacing): #Takes almost no time to init
        #convert to float and scale
        self.unscaled_patch=patch
        self.smIscale=patch.max()
        p0d=patch.astype(np.float)
        if self.smIscale>0:
            self.smI=p0d/self.smIscale##To not confuse with parent PolarMesh.I
        else:
            self.smI=p0d
        
        self.spacing=spacing
        c_est=np.array(patch.shape)//2
        c_est=c_est.astype(np.float)
        PolarMesh.__init__(self,cache,self.smI,self.smIscale,self.spacing,c_est)
        
    def update_plot(self):
#        for i in range(100):
#        print 'warning only doing iteration once'
        for i in range(1):
            self.do_cython_iter()
        mz,mx,my=self.update_mesh_coords()
#        self.do_iter()#without cython. Doesn't do eccentricity either

        self.small_mesh.mlab_source.set(x=mz,y=mx,z=my)
        o_mz,o_mx,o_my=self.apply_offset([mz,mx,my],self.offset)        
        self.whole_mesh.mlab_source.set(x=o_mz,y=o_mx,z=o_my)

#    def update_mesh_coords(self):
#        RadiusWrap=np.vstack([self.Radius,self.Radius[0,:][np.newaxis,:]])#Add theta at 2pi in addition to at 0 
#        mz= RadiusWrap*np.cos(self.phi_mesh) + self.Center[0]                         *self.spacing[0]
#        mx= RadiusWrap*np.sin(self.phi_mesh)*np.cos(self.theta_mesh) + self.Center[1] *self.spacing[1]
#        my= RadiusWrap*np.sin(self.phi_mesh)*np.sin(self.theta_mesh) + self.Center[2] *self.spacing[2]
#        return mz,mx,my

#####DEBUG#####
    def update_mesh_coords(self):
        RadiusWrap=np.vstack([self.Radius,self.Radius[0,:][np.newaxis,:]])#Add theta at 2pi in addition to at 0 
        mz= (RadiusWrap*np.cos(self.phi_mesh) + self.Center[0]                         )*self.spacing[0]
        mx= (RadiusWrap*np.sin(self.phi_mesh)*np.cos(self.theta_mesh) + self.Center[1] )*self.spacing[1]
        my= (RadiusWrap*np.sin(self.phi_mesh)*np.sin(self.theta_mesh) + self.Center[2] )*self.spacing[2]
        return mz,mx,my
        
    def draw_mesh_on_small_scene(self,scene):
        print 'WormSource2.py: draw_mesh_on_small_scene() fired'
        mz,mx,my=self.update_mesh_coords()
        self.small_mesh=mlab.mesh(mz,mx,my,representation='wireframe',color=(1,1,1),figure=scene.mayavi_scene)
            
    def draw_mesh_on_whole_worm(self,scene,offset):
        self.offset=offset*self.spacing
        mz,mx,my=self.update_mesh_coords()
        o_mz,o_mx,o_my=self.apply_offset([mz,mx,my],self.offset)
        self.whole_mesh=mlab.mesh(o_mz,o_mx,o_my,representation='wireframe',color=(1,1,1),figure=scene.mayavi_scene)
    
    def draw_mesh_on_scene_with_offset_without_update(self,scene,offset):
        self.offset=offset*self.spacing
        mz,mx,my=self.update_mesh_coords()
        o_mz,o_mx,o_my=self.apply_offset([mz,mx,my],self.offset)
        self.surrounding_mesh=mlab.mesh(o_mz,o_mx,o_my,representation='wireframe',color=(0,0,0),figure=scene.mayavi_scene)
        
    def apply_offset(self,coords,offset):
        c1,c2,c3=coords
        o1,o2,o3=offset
        return [c1+o1,c2+o2,c3+o3]

    def create_segmentation_image(self):
        self.bool_seg=self.return_segmented_image()#for documentation check multi_obj_polar_nucleus3D.py  (cython implementation)
        return self.bool_seg




class CubeMesh(HasTraits):
    dirty=Bool(False)#Consider making this an Event() which is write only
    flush_changes=Bool(False)
    
    floor_x_len=Int(1)
    floor_y_len=Int(1)
    floor_z_len=Int(1)
    floor_x_pos=Int(0)
    floor_y_pos=Int(0)
    floor_z_pos=Int(0)
    ceil_x_len=Int(1)
    ceil_y_len=Int(1)
    ceil_z_len=Int(1)
    ceil_x_pos=Int(1)
    ceil_y_pos=Int(1)
    ceil_z_pos=Int(1)
    x_len=Range('floor_x_len',high='ceil_x_len',enter_set=True,auto_set=False,mode='slider')#,editor=RangeEditor(low=1,high=100,mode='slider'))
    y_len=Range('floor_y_len',high='ceil_y_len',enter_set=True,auto_set=False,mode='slider')#,editor=RangeEditor(low=1,high=100,mode='slider'))
    z_len=Range('floor_z_len',high='ceil_z_len',enter_set=True,auto_set=False,mode='slider')
    x_pos=Range('floor_x_pos',high='ceil_x_pos',enter_set=True,auto_set=False,mode='slider')
    y_pos=Range('floor_y_pos',high='ceil_y_pos',enter_set=True,auto_set=False,mode='slider')
    z_pos=Range('floor_z_pos',high='ceil_z_pos',enter_set=True,auto_set=False,mode='slider')

    vol_module=Instance(Volume(),())
    
    def __init__(self,spacing=np.array([1.0,1.0,1.0])):     
        HasTraits.__init__(self)
    
        self.spacing=spacing
        self.array_src=ArraySource(spacing=self.spacing)
        self.unit_cube=np.mgrid[0:2,0:2,0:2]        
        self.unit_cube_vertices=self.unit_cube.reshape(-1,8).transpose()
        self.vertices=np.copy(self.unit_cube_vertices)
        self.sgrid=tvtk.StructuredGrid(dimensions=(2,2,2))
        self.sgrid.point_data.scalars=np.ones(self.vertices.shape)
        self.sgrid.point_data.scalars.name='cube_mesh_scalars'        
        self.update_vertices()
        self.grid_src=VTKDataSource(data=self.sgrid)

    @on_trait_change('dirty')##This is triggered after all coordinate changes are done.
    def update(self):
        if self.dirty:
#            print 'CubeMesh.update()'
            self.check_bounds()
            self.update_vertices()
            self.update_data()
#            print '   -resultant I.mean=',np.mean(self.I)
            self.dirty=False
            #calculate bounds()--implement

    def check_bounds(self):
#        if self.x_pos+self.x_len > self.window_shape[0]:
#            self.x_pos=self.window_shape[0]-self.x_len
#        if self.y_pos+self.y_len > self.window_shape[1]:
#            self.y_pos=self.window_shape[1]-self.y_len
#        if self.z_pos+self.z_len > self.window_shape[2]:
#            self.z_pos=self.window_shape[2]-self.z_len
#        if self.x_pos<0:self.x_pos=0
#        if self.y_pos<0:self.y_pos=0
#        if self.z_pos<0:self.z_pos=0
        if self.x_pos+self.x_len > self.image_shape[0]:
            self.x_pos=self.image_shape[0]-self.x_len
        if self.y_pos+self.y_len > self.image_shape[1]:
            self.y_pos=self.image_shape[1]-self.y_len
        if self.z_pos+self.z_len > self.image_shape[2]:
            self.z_pos=self.image_shape[2]-self.z_len
        if self.x_pos<0:self.x_pos=0
        if self.y_pos<0:self.y_pos=0
        if self.z_pos<0:self.z_pos=0
            
#    def observe_data(self,I,neuron_window_shape):
    def observe_data(self,worm,neuron_window_shape):
        if isinstance(worm,np.ndarray):
            self.I=worm
        else:            
            self.I=worm.array_src.scalar_data
        self.image_shape=self.I.shape
        self.window_shape=neuron_window_shape[::-1]
        self.set_slider_limits()
        self.update_vertices()#Make sure stays in bounds

    def set_slider_limits(self):
#        self.ceil_x_len=self.image_shape[0]-1#It looks like "xyz" but it's really "zyx"
#        self.ceil_y_len=self.image_shape[1]-1
#        self.ceil_z_len=self.image_shape[2]-1
#        self.ceil_x_pos=self.image_shape[0]-1
#        self.ceil_y_pos=self.image_shape[1]-1
#        self.ceil_z_pos=self.image_shape[2]-1
        self.ceil_x_len=self.image_shape[0]#It looks like "xyz" but it's really "zyx"
        self.ceil_y_len=self.image_shape[1]
        self.ceil_z_len=self.image_shape[2]
        self.ceil_x_pos=self.image_shape[0]
        self.ceil_y_pos=self.image_shape[1]
        self.ceil_z_pos=self.image_shape[2]
        self.floor_x_pos=0-self.window_shape[0]//2-1
        self.floor_y_pos=0-self.window_shape[1]//2-1
        self.floor_z_pos=0-self.window_shape[2]//2-1
        #Make starting guess
        self.x_len=self.window_shape[0]
        self.y_len=self.window_shape[1]
        self.z_len=self.window_shape[2]
        
    def display_grid_on_scene(self,scene):
        scene.mayavi_scene.add_child(self.grid_src)
        structured_grid_outline=StructuredGridOutline()
        self.grid_src.add_module(structured_grid_outline)
    


    def update_vertices(self):
        self.vertices=np.copy(self.unit_cube_vertices)
        self.lengths=[self.x_len,self.y_len,self.z_len]
        self.position=[self.x_pos,self.y_pos,self.z_pos]

        for d in range(3):
            self.vertices[:,d]*=self.lengths[d]
            self.vertices[:,d]+=self.position[d]
            
            if hasattr(self,'window_shape'):
                loc=self.vertices[:,d]>self.image_shape[d]
                self.vertices[loc,d]=self.image_shape[d]
            
        self.sgrid.points=self.spacing*self.vertices



    def display_volume_data_on_scene(self,scene):
        self.update_data()
        scene.mayavi_scene.add_child(self.array_src)
#        vol_module=Volume()
        self.volume=self.array_src.add_module(self.vol_module)
#        self.add_trait('volume',self.array_src.add_module(vol_module))
    def update_data(self): 
        self.unscaled_patch=self.I[self.x_pos:self.x_pos+self.x_len,#The use of "x" here is just symbolic
                               self.y_pos:self.y_pos+self.y_len,#More accurratly it is z,y,x
                               self.z_pos:self.z_pos+self.z_len]#Ex: I.shape=(201,1000,4200)
        self.unscaled_patch=self.unscaled_patch.astype(np.float)
        A=self.unscaled_patch-self.unscaled_patch.min() 
        M=A.max()
#        print 'WormSources2.py: cube.update_data() called'
#        print '    before we assign scalar_data'
        if M != 0:
#            print 'M is not 0'
            self.array_src.scalar_data=A/A.max()*255
#            self.array_src.scalar_data=temp/M*255
#            self.view_Image=temp/M*255
        else:
            self.array_src.scalar_data=np.zeros(A.shape)
#            print 'WormSources2.py:cube region has max == 0 '
#        self.view_Image=temp                               
                               
#        self.array_src.scalar_data=self.view_Image
        
#        print '    M',M
        
#        self.array_src.update()#debug





def calc_downsample(orig_shape,orig_stride,factor):
	orig_shape=np.array(orig_shape)
	orig_stride=np.array(orig_stride)
	new_stride=factor*orig_stride	
	new_shape=(np.int(a//factor) for a in orig_shape)

	return new_shape,new_stride

class MultiArray:
    def __init__(self,Array,downsample=1):#All should have same shape
        self.orig_Array=Array
        self.orig_shape=Array.shape
        self.orig_stride=Array.strides
        self.factor=downsample

    def set_downsample_to(self,factor):
        self.factor=factor
        self.view_Array=self.downsample_by(self.factor)
        
    def downsample_by(self,factor):
        new_shape,new_stride=calc_downsample(self.orig_shape,\
        self.orig_stride,factor)		
        self.sampled_Image = stride_tricks.as_strided( self.orig_Array, \
        shape=new_shape,strides=new_stride)
        return self.sampled_Image


class MultiVolume:	
    def __init__(self,Image,init_downsample=1,spacing=-1):#All should have same shape
        self.multiImage=MultiArray(Image)
        self.orig_shape=Image.shape
        self.orig_stride=Image.strides
        self.orig_spacing=np.array(spacing)
        
        self.factor=init_downsample
        
#        self.spacing=spacing
#        self.find_smallest_viable_factor()
        
        self.array=ArraySource()
        self.update()
        
    def display_on_scene(self,scene):
        self.volume=scene.mlab.pipeline.volume(self.array)

#    def set_downsample_to(self,factor):
    def update(self):
        print 'wholeworm downsample factor is',self.factor
        self.view_Image=self.multiImage.downsample_by(self.factor)
        print 'Resultant shape is',self.view_Image.shape,'\n'
        self.array.scalar_data=self.view_Image
        self.array.spacing=self.factor*self.orig_spacing
#        self.array.update()
#        self.volume.update()
    def reset_visualization_at_downsample(self,volume):
        volume.mlab_source.scalars=self.view_Image
        volume.mlab_source.reset(x=self.m_sIzi,y=self.m_sIxi,z=self.m_sIyi)
        
    def find_smallest_viable_factor(self):
        data_too_big=True
        iter=0
        while data_too_big:
#            print self.factor
            print 'iter'
            try:
                print 'try'
                if iter>3:
                    break
                self.view_Image=self.downsample_by(self.factor)
                Izi,Ixi,Iyi=np.indices(self.view_Image.shape)
                data_too_big=False
    #            break
            except:
                print 'except'
                self.factor*=2
                iter+=1
        else:
            self.sIzi,self.sIxi,self.sIyi=self.spacing[0]*Izi,self.spacing[1]*Ixi,self.spacing[2]*Iyi
            print 'necesary scale factor is ',self.factor
        
    def downsample_by(self,factor):
        new_shape,new_stride=calc_downsample(self.orig_shape,\
        self.orig_stride,factor)		
        self.sampled_Image = stride_tricks.as_strided( self.orig_Image, \
        shape=new_shape,strides=new_stride)
        return self.sampled_Image