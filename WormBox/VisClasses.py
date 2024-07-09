# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:16:37 2015

@author: csnyder
"""
from BoxCoords import BoxCoords
from BaseClasses import tWorm
import numpy as np
from mayavi.sources.api import ArraySource
from mayavi.modules.volume import Volume
from mayavi.modules.glyph import Glyph

from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.modules.structured_grid_outline import StructuredGridOutline
from tvtk.api import tvtk
from traits.api import on_trait_change,Property,Bool
import os
from traits.api import Int,Range,Instance,Str
import dill

def elementwise_max(arr1,arr2):
    return np.array([max(x1,x2) for x1,x2 in zip(arr1,arr2)])


def to_int(arr):
    return np.round(arr).astype(np.int)
def nearest_odd_int(arr):
    return np.ndarray.astype(1+2*np.round(arr//2),dtype=np.int)

def elementwise_max(arr1,arr2):
    return np.array([max(x1,x2) for x1,x2 in zip(arr1,arr2)])

def elementwise_min(arr1,arr2):
    return np.array([min(x1,x2) for x1,x2 in zip(arr1,arr2)])


class Box(BoxCoords):

    def __init__(self,window_shape,image_shape=None,must_have_odd_shape=True):
        if image_shape is None:
            image_shape=window_shape
        BoxCoords.__init__(self,window_shape,image_shape,must_have_odd_shape)
        self.setup_grid()
        self.grid_scenes=[]
    def setup_grid(self):
        self.unit_cube=np.mgrid[0:2,0:2,0:2]        
        self.unit_cube_vertices=self.unit_cube.reshape(-1,8).transpose()
        self.vertices=np.copy(self.unit_cube_vertices)
        self.sgrid=tvtk.StructuredGrid(dimensions=(2,2,2))
        self.sgrid.point_data.scalars=np.ones(self.vertices.shape)
        self.sgrid.point_data.scalars.name='cube_mesh_scalars'        
        self.update_vertices()
        self.grid_src=VTKDataSource(data=self.sgrid)

    def display_grid_on_scene(self,scene,offset=None):
        self.grid_scenes.append(scene)
        scene.mayavi_scene.add_child(self.grid_src)
        structured_grid_outline=StructuredGridOutline()
        self.grid_src.add_module(structured_grid_outline)

#    @on_trait_change('center.[x,y,z],radius.[x,y,z]')
    @on_trait_change('pos.[x,y,z],length.[x,y,z]')
    def update_vertices(self):
        self.vertices=np.copy(self.unit_cube_vertices)
        for d in range(3):
            self.vertices[:,d]*=self.length[d]
            self.vertices[:,d]+=self.pos[d]
        self.sgrid.points=self.spacing*self.vertices



from mayavi import mlab
from WormSources2 import SphereMesh
from BaseClasses import Segments
from Nucleus import PolarCache
from WormSources2 import Neuron
from traits.api import HasTraits
#class VisSegments(Segments):
class VisSegments(HasTraits):
#    worm=Instance(tWorm,())
    def __init__(self,nm_window_shape,smear):
        self._dict=Segments()
        self.nm_window_shape=nm_window_shape
        self.smear=smear
        
#This method sets the pixel_window_shape within the context of the
##The worm image data we're actually going to use to segment stuff
    def set_data(self,channel):
        self.seg_channel=channel
        self.spacing=self.seg_channel.spacing
        self.nm_voxel_shape=self.seg_channel.nm_voxel_shape
        self.window_shape=nearest_odd_int(self.nm_window_shape/self.nm_voxel_shape)
        self.polar_cache=PolarCache(self.window_shape,self.spacing,self.smear)

    def unscaled_patch(self,point=None):
        #point will default to worm.center if left None
        return self.seg_channel.get_unscaled_padded_patch(point,self.worm.window_radius)#enforce to have output shape window_shape

    def draw_centers_on_scene(self):
        self.points=np.vstack([v['marker'] for v in self.values()])
        self.nodes=mlab.points3d(self.points[0],self.points[1],self.points[2],name='new marker',color=(1,0,0))         
        self.nodes.glyph.glyph.scale_factor=5

    def create_mesh_if_not_hasattr(self,key):
        if not 'mesh' in self[key].keys():
            nm_point=self[key]['marker']
            point=nm_point//self.seg_channel.nm_voxel_shape
            Patch=self.seg_channel.get_window_sized_unscaled_padded_patch(point=point)
            self[key]['mesh']=SphereMesh(self.polar_cache,Patch,spacing=self.spacing)

#    def segment_this_neuron(self,key):
#        self.create_mesh_if_not_hasattr(key)

    def draw_mesh_on_scene(self,key=None,scene=None):
        if key is None:
            key=self.current_name
        self.create_mesh_if_not_hasattr(key)
        
        offset=np.array([0.0,0.0,0.0])
#        offset=self.seg_channel.origin

        scene=self.seg_channel.small_scene
        self[key]['mesh'].draw_mesh_on_small_scene(scene,offset)
        
        
#        for scene in self.seg_channel.drawn_scenes:
#            if scene.is_local_view:


#    def draw_list_of_meshes(self,names):#Just code small scene for now
#        if not hasattr(self.seg_channel,'small_scene'):
#            print 'no attr small_scene'
#            return
#        scene=self.seg_channel.small_scene
#        offset=self.seg_channel.origin
#        for key in names:
#            if 'mesh' in self[key].keys():
#                self[key]['mesh']
#                
#            for scene in self.drawn_scenes:#Counts each scene once
#                self.draw_neuron_on_scene(nm_point,scene,label=name)
                
                
###FROM BEFORE
#        print 'update_region fired'
#        self.current_neuron=self.segments[self.neuron_name]
#        self.update_box()
#        if not self.current_neuron.has_key('mesh'):
#            self.current_neuron['is_segmented']=True ##A place holder right now. It means user is satisfied with this neuron segmentation
#            Patch=self.cube.unscaled_patch
#            print 'update_region: unscaled_patch.mean():',Patch.mean()
##            Patch=self.cube.array_src.scalar_data
#            self.current_neuron['mesh']=SphereMesh(self.polar_cache,Patch,spacing=self.spacing)
#            offset=self.current_neuron['marker'][::-1] - self.window_radius[::-1]
#            self.current_neuron['mesh'].draw_mesh_on_whole_worm(self.whole_worm,offset)
##        if self.record_gui:
##            if not self.current_neuron.has_key('record'):
##                self.current_neuron['record']=GUIRecord(self.env,self.file_name)
##            self.current_record=self.current_neuron['record']
#        self.current_mesh=self.segments[self.neuron_name]['mesh']##softcopy that should bind them.
#        self.reset_params()#set the traits on the GUI to reflect the active mesh
#        self.current_mesh.draw_mesh_on_small_scene(self.small_scene)
#
#        try:
#            self.marker_status=self.current_neuron['notes']['quality']
#        except:
#            self.marker_status=self.marker_status_options[0]
#
#        self.meshes_in_view=list_meshes_in_cube(self.cube,self.neuron_name,self.segments)
#        self.surrounding_mesh_list=[]
#        for key,offset in self.meshes_in_view:
#            self.segments[key]['mesh'].draw_mesh_on_scene_with_offset_without_update(self.small_scene,offset)
#            self.surrounding_mesh_list.append(self.segments[key]['mesh'])
#        self.set_mesh_visibility()

#############Some stuff for shifting segment markers ########
#s='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos3_248_233_len11_157_251/Results/Segments.pkl'
#import dill
#with open(s,'rb') as handle:
#    S=dill.load(handle)
#nm_voxel_shape=gui.worm['mba30'].nm_voxel_shape
#inc=nm_voxel_shape*np.array([1.0,0.0,0.0])*3
#for seg in S.values():
#    seg['marker']+=inc
#f2='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos-3_248_233_len23_157_251/Results/'
#s2=f2+'Segments.pkl'
#with open(s2,'wb') as handle:
#    dill.dump(S,handle)


#HACK DEBUG sorry..
    def __getattr__(self,attr):
        try:
            return getattr(self,attr)
        except:
            return getattr(self._dict,attr)

    def __getitem__(self,item):
        try:
            return self[item]
        except:
            return self._dict[item]
    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return self._dict.__repr__()

from traits.api import Array
from traits.api import Instance
from BaseClasses import ApoReader

from BaseClasses import tEnvironment

class MultiChannelWorm(BoxCoords):#Doesn't handle plotting, so inheriting BoxCoords instead is fine
    
    downsample=Int(1)
    
#    window_shape=Array()#In nanometers
    
#    spacing=Array()#defined in tWorm
#    smear=Instance(Array,([1.0,1.0,1.0]))#registering as list bleh
    t=Int(0)    
    vol_module=Instance(Volume,())
    
    name=Str('')
    results_dir=Str('')
    segments_filename=Str('Segments.pkl')#Default save name
    cache_dir=''#fix later with pickle mixin
    
#    def __init__(self,window_shape=None,spacing=None,smear=None,*args):
    def __init__(self,nm_window_shape,*args):        
        self.nm_window_shape=nm_window_shape        
        
        self.string_args=[]
        for arg in args:
            if isinstance(arg,np.ndarray):
                self.nm_image_shape=arg
            elif isinstance(arg,str):
                self.string_args.append(arg)
                
        if len(self.string_args)>0:
            self.env=tEnvironment(*self.string_args)##Get nm shape for FOV
            self.nm_image_shape=self.env.nm_image_shape
                

#        if len(args) is not 0:###Broken. causes misalignment of grid and image
#        else:
#            self.nm_image_shape=self.nm_window_shape#best guess for now: updated in add_channel

        
        self.smear=np.array([1.0,1.0,1.0])
        
        self.channels=dict() #channels is an aux
        
        BoxCoords.__init__(self,self.nm_window_shape.astype(np.int),self.nm_image_shape.astype(np.int),must_have_odd_shape=False)##Box keeps things in units of nm
        self.segments=VisSegments(self.nm_window_shape,self.smear)

#        self.segments_filename='Segments.pkl'#default name for saving segments class

    def add_channel(self,name,Env,spacing=None):   
        
        print 'Configuring Channel:,',name
        
        worm=Channel(self,Env,self.nm_window_shape,spacing)
#        worm=tWorm(self,Env)
#        worm.segments=self.segments#same segments for each channel #Probably fix later
        worm.dask_read_data()
        self.channels[name]=worm
        worm.name=name
        
#        window_bdrys=elementwise_max(self.nm_image_shape,worm.nm_image_shape)
#        self.nm_image_shape=window_bdrys.astype(np.int)
#        self.image_shape=self.nm_image_shape

    def add_channel_directly(self,name,*args):
        chan=Channel(self,self.nm_window_shape,*args)
        chan.name=name
        self.channels[name]=chan


    def add_channel_derived_from(self,name,with_process,named):
        process=with_process;new_name=named
        print 'Configuring Channel,',new_name
        original=self.channels[name]
        env=original#channel inherits env.This will turn out just like passing the env portion
        worm=Channel(original.main_worm,env,original.nm_window_shape)

        cache_file=self.cache_dir+new_name+'.npy'
        if os.path.isfile(cache_file):
            worm.X = np.load(cache_file)#read from cache
            print 'warning, ignoring generating process for channel:',new_name
            print '...reading from cache instead'
        else:
            print 'Generating channel data denovo...'
            worm.X=process(original.X)
            print 'Caching results for future...'
            if isinstance(worm.X,np.ndarray):
                np.save(cache_file,worm.X)#Assuming this is an np array for now

        self.channels[new_name]=worm

    def segmentation_channel_is(self,name):
        self.segments.set_data(self.channels[name])
    def extract_time_series_data_for(self,name):
        self.time_series_data=self.channels[name]
        
    def __getitem__(self,item):
#        if isinstance(item,str):
#            return self.channels[item]
        try:
            return self.channels[item]
        except:
            raise Exception('must be channel name')

    def new_neuron_at(self,nm_point,label=None):
#        if label is None:
#            label=str(len(self.segments._dict.items()))
#        self.segments._dict[label]=dict(marker=nm_point)
        self.segments.new_neuron_at(nm_point)
        
        for channel in self.channels.values():
            for scene in channel.drawn_scenes:#Counts each scene once
                channel.draw_neuron_on_scene(nm_point,scene)
    def draw_neuron_at(self,nm_point,label=None):
        for channel in self.channels.values():
            for scene in channel.drawn_scenes:#Counts each scene once
                channel.draw_neuron_on_scene(nm_point,scene)
 
    def remove_segments_from_small_views(self):
        for channel in self.channels.values():
            for scene in channel.drawn_scenes:#Counts each scene once
                if scene.is_local_view:
                    MScene=scene.mayavi_scene
                    for child in MScene.children:
                        if isinstance(child,Neuron):
                            MScene.remove_child(child)
                        
    def get_segments_in_range(self):#buggy#no idea#Sometimes doesn't plot stuff that is in range
        in_range_names=[]
        in_range_segs=[]

        for name,seg in zip(self.segments._dict.keys(),self.segments._dict.values()):
            if self.in_range(seg['marker']):###Has to be in nanometers
                in_range_names.append(name)
                in_range_segs.append(seg)
        return in_range_names,in_range_segs
                

    @on_trait_change('center.[z,y,x],radius.[z,y,x]')
    def refresh_small_scenes(self):
        if len(self.segments) == 0:
            return
        print 'VisClasses214 refresh_small_scene'
        names,segments=self.get_segments_in_range()
        for name,segm in zip(names,segments):
#            self.new_neuron_at(segm['marker'],label=name)
            self.draw_neuron_at(segm['marker'],label=name)

#            self.segments.draw_list_of_meshes(names)

    def save_segments(self,fname=None):
        if not os.path.isdir(self.results_dir):
            print 'worm.results_dir is not a recognized directory...'
            print '    ...aborting save'
            return           
        f=os.path.join(self.results_dir,self.segments_filename)     
        with open(f,'wb') as handle:
            print 'saving segments...'
#            dill.dump(self.segments,handle)
            dill.dump(self.segments._dict,handle)
            print '    ...complete'

    @on_trait_change('results_dir, segments_filename')
    def load_segments(self,fname=None):
        if not os.path.isdir(self.results_dir):
            print 'worm.results_dir is not a recognized directory'
            return
        else:
            f=os.path.join(self.results_dir,self.segments_filename)
            if not os.path.isfile(f):
                print 'warning.. ,',f,'.. is not a recognized file'
            else:
                print 'Found previous session data:',f
                print '    ...loading previous data'

                with open(f,'r') as handle:
                    self.segments._dict=dill.load(handle)
                print '    ...complete'

class Channel(tWorm,Box):
#    spacing=Array()#defined in tWorm
#    smear=Instance(Array,([1.0,1.0,1.0]))#registering as list bleh

    downsample=Int(1)
    
#    max_time_step=Int(1)
#    min_time_step=Int(0)
#    time_step=Range(low='min_time_step',high='max_time_step',value=0)
    time_step=Int(0)    
    
    
    vol_module=Instance(Volume,())
    main_worm=Instance(MultiChannelWorm,())
#    segments=VisSegments

    has_time_component=Bool(False)

    name=Str('')

    small_vol_module=Instance(Volume(),())

    origin=Array()    
    
#    def __init__(self,main_worm,Env,nm_window_shape,spacing=None):
    def __init__(self,*args):
        tWorm.__init__(self)
        self.neuron_glyphs=dict()
        
        for arg in args:
            if isinstance(arg,tEnvironment):#environment class
                self.read_dir=arg.read_dir
                self.write_dir=arg.write_dir
#                tEnvironment.__init__(self,arg.read_dir,arg.write_dir)
            if isinstance(arg,MultiChannelWorm):
                self.main_worm=arg
            if isinstance(arg,str):#read_dir
                if os.path.isfile(arg):
                    if self.is_ini(arg):
                        self.set_ini_file(arg)
                    else:
                        print 'a file was passed that was not an .ini file'
                else:
                    self.read_dir=arg
#                tEnvironment.__init__(self,arg)
            if isinstance(arg,np.ndarray):#numpy stack
                if np.min(arg)==1:
                    self.spacing=arg
                else:
                    self.nm_window_shape=arg
            if isinstance(arg,tWorm):
                self.template_worm=arg

        if hasattr(self,'template_worm'):
            if self.template_worm._ini_file!='':
                self.set_ini_file(self.template_worm._ini_file)
            else:
                if hasattr(self.template_worm,'nm_voxel_shape'):
                    self.nm_voxel_shape=self.template_worm.nm_voxel_shape
                if self.template_worm.spacing.max()>1:
                    self.spacing=self.template_worm.spacing
            if hasattr(self.template_worm,'npX'):
                self.npX=self.template_worm.npX
            if hasattr(self.template_worm,'daX'):
                self.daX=self.template_worm.daX

        if hasattr(self.X,'shape'):
            self.image_shape=self.X.shape[-3:]
            self.nm_image_shape=self.image_shape*self.nm_voxel_shape
        else:
            self.image_shape=to_int(self.nm_image_shape/self.nm_voxel_shape)#DEBUG# this may not agree with X.shape



        self.window_shape=nearest_odd_int(self.nm_window_shape/self.nm_voxel_shape)

        
        #######

        self.main_worm.center.upr_bdd_is_atleast(self.nm_image_shape)#Allow center to transverse largest channel

        Box.__init__(self,self.window_shape,self.image_shape)
        
#        self.image_shape=np.array(self.X.shape)
        
        ###I think spacing is not currently being handled
#        if self._ini_file=='':
#            if spacing is not None:           
#                self.spacing=spacing
#            else:
#                print 'warning, no way to determine pixel spacing...assuming uniform'
        
        self.nm_spacing=self.spacing*self.nm_voxel_shape
#        if smear is not None:
#            self.smear=smear
#        else:
        self.smear=np.array([1.0,1.0,1.0])#Fix later        
        self.drawn_scenes=[]
        self.vol_modules=[]

        self.small_vol_module=Volume()
#        self.add_trait('small_vol_module',Volume())
#        self.vol_modules.append('small_vol_module')





    def clear_mesh(self):
        if not hasattr(self,'small_scene'):
            return
#        MScene=self.current_mesh.small_mesh.scene.mayavi_scene
#        MScene.remove_child(self.current_mesh.small_mesh)
        scene=self.small_scene
        MScene=scene.mayavi_scene
        for child in MScene.children:
            if child.name=='GridSource':
                MScene.remove_child(child)
                
                
        
    def add_neuron(self,point,add_offset,label=None):
        point=np.array(point)
        print 'Channel: raw point added',point
        point/=self.spacing
        if add_offset:
            point+=self.center.__array__()
#            point+=self.origin
        print 'Channel: point+offset:',point
        print '    *nm_vox',self.nm_voxel_shape * point
        self.main_worm.new_neuron_at(self.nm_voxel_shape * point)


        
    def draw_neuron_on_scene(self,nm_center,scene):
        center=nm_center//self.nm_voxel_shape
        #Check if is in view
        if scene.is_local_view:
            if not self.in_range(center):#abort if we're a local view and can't see the thing
#                print 'Channel258: added neuron not in range'
#                print '     channel:',self.name
#                print '     scene:', scene.name
                return
        
        if scene.is_local_view:
#            center -= np.array( self.pos - self.origin) 
            center -= self.center.__array__()

#        print 'center added251:',center
        center*=self.spacing
#        print '    *spacing:',center

        marker_sphere=Neuron(center)
#        self.neuron_glyphs[label]=marker_sphere#actually broken when 2xChannel for 1xScene
        marker_sphere.data_source.radius=1500/np.mean(self.nm_voxel_shape)
        scene.mayavi_scene.add_child(marker_sphere)
        marker_sphere.surface.actor.property.color=(1.0,0.0,0.0)#only works if called afterward
    

    def get_unscaled_padded_patch(self,point=None,radius_size=None,time_step=None):

        if radius_size is None:
            radius_size=self.radius
        if point is None:
            point=self.center.__array__()
        point=np.array(point)
        zero=np.zeros_like(point)
        upr=elementwise_min(self.image_shape,point+radius_size+1)
        lwr=elementwise_max(zero,point-radius_size)
        upad=elementwise_max(zero, -self.image_shape+(point+radius_size+1)  )
        lpad=elementwise_max(zero,-point+radius_size)
        self.padd=tuple([(l,u) for l,u in zip(lpad,upad)])

        if self.X.ndim is 4:
            if time_step is None:
                tt=self.time_step
            else:
                tt=time_step
#            dst=self.X[tt]
            patch=np.array(self.X[tt,lwr[0]:upr[0],lwr[1]:upr[1],lwr[2]:upr[2]])
        else:
#            dst=self.X
            patch=np.array(self.X[lwr[0]:upr[0],lwr[1]:upr[1],lwr[2]:upr[2]])

        if self.padd != ((0,0),(0,0),(0,0)):
            if patch.ndim is 3:
                padded_patch=np.pad(patch,self.padd,mode='constant',constant_values=0)
            elif patch.ndim is 4:##This happens in a special case where we're looking at alot of timepoints at once
                padd=((0,0),self.padd[0],self.padd[1],self.padd[2])
                padded_patch=np.pad(patch,padd,mode='constant',constant_values=0)
        else:
            padded_patch=patch
        return padded_patch
        
    def get_window_sized_unscaled_padded_patch(self,point=None,time_step=None):
        return self.get_unscaled_padded_patch(point,self.window_radius,time_step)
    
    def scale_data(self,arr):
        d_arr=arr.astype(np.float)
        A=d_arr-d_arr.min()
        M=A.max()
        if M==0:
            dst=arr
            scale=-1
        else:
            dst=255*A/M
            scale=M/255.0
        return dst,scale


    def draw_on_scene(self,scene,*args):
        
        if hasattr(scene,'time_step'):
            self.sync_trait('time_step',scene,'time_step',mutual=True)
            scene.max_time_step=self.X.shape[0]-1
#            self.max_time_step=self.X.shape[0]
        
#    def draw_on_scene(self,scene,module):
        self.drawn_scenes.append(scene)
        ##Define grid drawing
        
        scene.worm=self
        if scene.is_local_view:
            self.display_on_small_scene(scene,*args)
#            self.display_on_small_scene(scene,module)
        else:
#            scene.sync_trait('downsample',self,mutual=True)
            self.display_on_large_scene(scene,*args)
#            self.display_on_large_scene(scene,module)

#worm['cm1'].small_array_src
#worm['cm1'].small_vol_module.actors[0].print_traits()  #(origin) offset  
    def set_small_scene_patch(self):
        if not hasattr(self,'small_array_src'):
            return
        _patch=self.get_unscaled_padded_patch()
        patch,self.small_scene_scale=self.scale_data(_patch)
        self.small_array_src.scalar_data=patch
#        self.small_array_src.origin=-self.radius.__array__()        
#    def display_on_small_scene(self,scene,module):#no option for downsample at the moment
    
    @on_trait_change('time_step')
    def reset_small_and_large_patch(self):
        print 'time_step changed'
        self.set_small_scene_patch()
        if not hasattr(self,'large_array_src'):
            return
        if self.X.ndim is 4:
            dst=self.X[self.time_step]
        else:
            dst=self.X
        f=self.large_scene_downsample
        patch,self.large_scene_scale=self.scale_data(np.array(dst[:,::f,::f]))#z,y,x 
        self.large_array_src.scalar_data=patch
        
    def redraw_neurons(self,force_all=False):
        if len(self.main_worm.segments) == 0:
            return
        for scene in self.drawn_scenes:
            if scene.is_local_view and not force_all:
                self.remove_segments_from_scene(scene)

        names,segments=self.main_worm.get_segments_in_range()
        for name,seg in zip(names,segments):
            nm_point=seg['marker']
            for scene in self.drawn_scenes:#Counts each scene once
                self.draw_neuron_on_scene(nm_point,scene)
            
    def remove_segments_from_scene(self,scene):
        MScene=scene.mayavi_scene
        neurons=[]
        for child in MScene.children:
            if isinstance(child,Neuron):
                neurons.append(child)
#        print '491remove_segments:len:',len(neurons)
        
        for neuron in neurons:
                MScene.remove_child(neuron)
    
        for child in MScene.children:
            if child.name=='small sphere mesh':
                MScene.remove_child(child)

    def display_on_small_scene(self,scene,*args):#no option for downsample at the moment
#        self.small_array_src=ArraySource(spacing=self.nm_spacing)
#        self.small_array_src=ArraySource(spacing=self.spacing,origin=( self.center.__array__() - self.pos.__array__()) )
        self.set_origin()

        print 'origin is',self.origin
#        self.small_array_src=ArraySource(spacing=self.spacing,origin=( -self.radius.__array__() ) )
        self.small_array_src=ArraySource(spacing=self.spacing,origin=self.origin)
#        self.small_array_src=ArraySource(spacing=self.spacing)
        self.small_scene=scene#for debug
        self.set_small_scene_patch()
        scene.mayavi_scene.add_child(self.small_array_src)
        self.small_volume=self.small_array_src.add_module(self.small_vol_module)
        scene.draw_axes(self.small_array_src.scalar_data.shape)
        print 'data shape is',self.small_array_src.scalar_data.shape
        
#        self.small_volume=self.small_array_src.add_module(module)
#    def display_on_large_scene(self,scene,module):#entire view, possibly scaled down
        
    def display_on_large_scene(self,scene,*args):#entire view, possibly scaled down
        print 'visclasses296 large_scene_display'
        #let downsample stay just on the scene
        #Because conceivably the same channel might be not downsampled in another scene
        f=np.abs(scene.downsample)#take every fth element #just in x,y planes#abs is a hack bc can be -1 if uninit
        self.large_scene_downsample=f
        if self.X.ndim is 4:
            dst=self.X[self.time_step]
        else:
            dst=self.X

        downsample_mat=np.array([1.0,f,f])
        patch,self.large_scene_scale=self.scale_data(np.array(dst[:,::f,::f]))#z,y,x
        
        self.large_array_src=ArraySource(spacing=self.spacing*downsample_mat)
#        self.large_array_src=ArraySource(spacing=self.nm_spacing*downsample_mat)      
        ####DEBUG###
        #What to do if you want to change the downsampling?#
#        self.on_trait_change('scene.downsample',
        self.large_array_src.scalar_data=patch
        scene.mayavi_scene.add_child(self.large_array_src)
#        self.large_vol_module=Volume()
        self.add_trait('large_vol_module',Volume())
        self.vol_modules.append('large_vol_module')
#        self.large_volume=self.large_array_src.add_module(self.large_vol_module)        
        
        scene.mayavi_scene.add_child(self.large_array_src)
#        self.large_volume=self.large_array_src.add_module(module)
        self.large_volume=self.large_array_src.add_module(self.large_vol_module)        

#        self.large_volume=self.large_array_src.add_module(self.vol_module)


    @on_trait_change('main_worm.center.[z,y,x]')##Could be made simpler with Property setter I think
    def update_center(self):
        self.center.set_value(self.main_worm.center.__array__() // self.nm_voxel_shape)
        self.set_small_scene_patch()
        self.redraw_neurons()
        self.clear_mesh()
        
    @on_trait_change('main_worm.radius.[z,y,x]')
    def update_radius(self):
        self.radius.set_value(self.main_worm.radius.__array__() // self.nm_voxel_shape)
        self.set_small_scene_patch()
        self.set_origin()
        if hasattr(self,'small_array_src'):
            self.small_array_src.origin=self.origin
#            self.small_array_src.origin=-self.radius.__array__()
        self.redraw_neurons()
        for scene in self.drawn_scenes:
            scene.update_axes()
            
    @on_trait_change('main_worm.pos.[z,y,x]')
    def update_pos(self):
        self.pos.set_value(self.main_worm.pos.__array__() // self.nm_voxel_shape)
    @on_trait_change('main_worm.length.[z,y,x]')
    def update_length(self):
        self.length.set_value(self.main_worm.length.__array__() // self.nm_voxel_shape)


    def set_origin(self):
#        self.origin=-np.floor((self.radius.__array__())*self.spacing)
        self.origin=-1*self.radius.__array__()*self.spacing
        
        
        
        
