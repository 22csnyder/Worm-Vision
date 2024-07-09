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



import os
class GUIRecord:
    __version__=2.0 #Vis class may change in the future
    
    event_list=[]
    
    def __init__(self,env,apo):
        self.env=env #Keeps track of directory and data this corresp to
        self.apo=apo #Keeps track of what apo file we're using to number markers
        self.record_write_dir=self.env.write_dir+'GUIRecordings/'
        
        try:
            os.makedirs(self.env.write_dir)#Make 
        except OSError:
            print 'warning could not make write_dir'
#            if not os.path.isdir(path):
#                raise



class Segment:###Don't Change this class or risk not being able to open the .pkl
    def __init__(self,name,number,offset,binary_patch,float_patch,notes=None):
        self.name=name
        self.number=number
        self.offset=offset
        self.binary_patch=binary_patch
        self.float_patch=float_patch
        if notes is None:
            self.notes=dict(quality='Good')
        if isinstance(notes,dict):
            self.notes=notes
        
def list_meshes_in_cube(cube,center_neuron,segments):
    meshes_in_view=[]
    center_main=np.array(segments[center_neuron]['marker'][::-1])
    for key in segments.keys():
        if segments[key].has_key('mesh') and key is not center_neuron:#initialized meshes besides the current one
            center_other=np.array(segments[key]['marker'][::-1])
            xl=cube.x_pos<center_other[0]<cube.x_pos+cube.x_len
            yl=cube.y_pos<center_other[1]<cube.y_pos+cube.y_len
            zl=cube.z_pos<center_other[2]<cube.z_pos+cube.z_len
            if xl and yl and zl:
                offset=center_other-center_main
                meshes_in_view.append([key,offset])
    return meshes_in_view
    
def cbk(vtk_obj,event):
    print vtk_obj.GetClassName(),event,vtk_obj.GetKeyCode()
    #use tvtk.to_tvtk(vtk_obj) to get the tvtk version of this object
    
def ButtonEvent(obj,event):
    if event== "RightButtonPressEvent":
        print 'right button was pressed'
	        

#class CubeCoords(HasTraits):
#    dirty=Bool(False)#Consider making this an Event() which is write only
#    flush_changes=Bool(False)
#    x_len=Range(1,100,value=34)
#    y_len=Range(1,500,value=74)
#    z_len=Range(1,500,value=74)
#    x_pos=Int()
#    y_pos=Int()
#    z_pos=Int()
#    vol_module=Instance(Volume(),())
#    
#    def __init__(self):
#        HasTraits.__init__(self)

#from Do_sphharm_segmentations import SlideShow

def get_Patch(center,volume,window_shape):
    window_radius=np.array(window_shape)//2
    x_pos=center[0]-window_radius[0]
    y_pos=center[1]-window_radius[1]
    z_pos=center[2]-window_radius[2]
    x_len,y_len,z_len=window_shape
#    print 'z_pos:',z_pos
#    print 'z_len:',z_len
    patch=volume[x_pos:x_pos+x_len+1,
               y_pos:y_pos+y_len+1,
               z_pos:z_pos+z_len+1]
    return patch

class Vis(HasTraits):
#    _cube=Instance(CubeCoords,())
#    cube=Instance(CubeMesh,())
    record_gui=Bool(True)#Don't turn this off then on in the middle of the program
    
    max_time=Int(1)
    min_time=Int(0)
    time=Range('min_time','max_time',mode='slider')
    
    do_global_iteration=Button()
    next_iteration=Button()#do iteration
    reset_current_mesh=Button('retry')
    
    Average_Intensity=Float()
    
    radius_best_guess=Range(low=1.0,high=15.0,)        
    emphasis_on_radius_guess=Range(low=0.0,high=30.0)
    smoothness=Range(low=0.0,high=500.0)     
    emphasis_on_marker_guess=Range(low=0.0,high=500.0)
    
    toggle_mesh_visibility=Button('on/off')
    
    neuron_name=Str('')
    m=Int(-1,record=True)
#    n_ind=Int(-1)
    
    next_marker=Button()
    previous_marker=Button()
    save_session=Button() ##Button as of yet unimplemented
    save_at_end=Bool(False)    
    
    down=Button('sample') 
    up=Button('sample')

    forward=Button('+')
    backward=Button('-')

    debug=Button('debug')    
    
    to_var=Button()
    to_normal=Button()
    
    engine1 = Instance(Engine, args=())
    engine2 = Instance(Engine, args=())
    whole_worm      = Instance(MlabSceneModel,())
    small_scene     = Instance(MlabSceneModel,())
    
    mesh_visibility=Bool(True)
    
    current_sphere_mesh= Instance(SphereMesh)
    
#    integral=Float(1.2)

#    cube=Instance(CubeMesh(),())
    
    volume_control=Instance(Volume,())

    cube=Instance(CubeMesh,())#Either seem to work

    box_x_len=DelegatesTo(delegate='cube',prefix='x_len')
    box_y_len=DelegatesTo(delegate='cube',prefix='y_len')
    box_z_len=DelegatesTo(delegate='cube',prefix='z_len')
    box_x_pos=DelegatesTo(delegate='cube',prefix='x_pos')
    box_y_pos=DelegatesTo(delegate='cube',prefix='y_pos')
    box_z_pos=DelegatesTo(delegate='cube',prefix='z_pos')
    
    floor_x_len=DelegatesTo('cube')
    floor_y_len=DelegatesTo('cube')
    floor_z_len=DelegatesTo('cube')
    floor_x_pos=DelegatesTo('cube')
    floor_y_pos=DelegatesTo('cube')
    floor_z_pos=DelegatesTo('cube')

    ceil_x_len=DelegatesTo('cube')
    ceil_y_len=DelegatesTo('cube')
    ceil_z_len=DelegatesTo('cube')
    ceil_x_pos=DelegatesTo('cube')
    ceil_y_pos=DelegatesTo('cube')
    ceil_z_pos=DelegatesTo('cube')     
    
#    cube_display=VGroup('box_coords.x_len','box_coords.y_len','box_coords.z_len','box_coords.x_pos','box_coords.y_pos','box_coords.z_pos')
    
    bad_marker=Button('mark as bad')
    marker_status_options=['','marker is bad']
    marker_status=Enum(marker_status_options)
    
    init_is_finished=Bool(False)
    save_status_options=['session will not save at end','session will save at end']
    save_status=Enum('session will not save at end',['session will not save at end','session will save at end'])
    
    show_right_panel=Button('color control');display_right_panel=Bool(False)
    show_global_view=Button('global view');display_global_view=Bool(False)
    
    def _whole_worm_default(self):
        self.engine1.start()
        return MlabSceneModel(engine=self.engine1)

    def _small_scene_default(self):
        self.engine2.start()
        return MlabSceneModel(engine=self.engine2)
        
    def __init__(self,worm_class_instance,env=None,spacing=None,smear=None,neuron_window_shape=None):#nws in x,y,z format
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        
        self.segments=collections.OrderedDict()
        self.env=env

        if isinstance(worm_class_instance,np.ndarray):#first argument is a patch to use
            self.Patch=worm_class_instance            
            self.neuron_window_shape=np.array(self.Patch.shape[::-1])#Please be odd
            self.add_trait('spacing',Array(value=spacing))
            self.cube=CubeMesh(self.spacing)
            self.sync_trait('volume_control',self.cube,'vol_module',mutual=True)
            self.cube.observe_data(self.Patch,self.neuron_window_shape)
            self.display_global_view=False
            self.cube.display_volume_data_on_scene(self.small_scene)
            self.window_radius=np.array((self.neuron_window_shape-1)//2)#x,y,z        
            self.patch_shape=2*self.window_radius[::-1]+1#z,y,x
            self.polar_cache=PolarCache(self.patch_shape,self.spacing)#x,y,z patch, but z,y,x spacing
            self.segments['0']=dict(marker=self.window_radius+1)
            self.file_name='nothing in particular'
            self._next_marker_fired()
            print 'vis is init'
            self.init_is_finished=True
        else:
            self.init_is_finished=True
                
            self.neuron_window_shape=neuron_window_shape#x,y,z
    #        self.spacing=spacing
            self.add_trait('spacing',Array(value=spacing))
            
            
            
            self.worm=worm_class_instance
            self.max_time=self.worm.daX.shape[0]-1        
            self.worm.spacing=spacing
            self.worm.display_on_scene( self.whole_worm )
                    
            self.cube=CubeMesh(self.spacing)
            
            self.sync_trait('volume_control',self.cube,'vol_module',mutual=True)
    #        self.cube.observe_data(self.multivolume.multiImage.orig_Array)
    #        self.cube.observe_data(self.worm.array_src.scalar_data,self.neuron_window_shape)
            self.cube.observe_data(self.worm,self.neuron_window_shape)
            self.cube.display_grid_on_scene(self.whole_worm)
            self.cube.display_volume_data_on_scene(self.small_scene)
     
            
    #        neuron_window=np.array([75,75,35])#must be odd #Falsh
            self.window_radius=np.array((self.neuron_window_shape-1)//2)#x,y,z        
            self.patch_shape=2*self.window_radius[::-1]+1#z,y,x
            self.polar_cache=PolarCache(self.patch_shape,self.spacing)#x,y,z patch, but z,y,x spacing
            
            print 'vis is init'
            
            self.init_is_finished=True

    @on_trait_change('radius_best_guess')
    def update_rad(self):
        self.current_mesh.r_est=self.radius_best_guess
        if self.record_gui:
            self.current_record.event_list.append(('radius_best_guess',self.radius_best_guess))
    @on_trait_change('emphasis_on_radius_guess')
    def update_lamR(self):
        self.current_mesh.lamR=self.emphasis_on_radius_guess
        if self.record_gui:
            self.current_record.event_list.append(('emphasis_on_radius_guess',self.emphasis_on_radius_guess))
    @on_trait_change('emphasis_on_marker_guess')
    def update_lamM(self):
        self.current_mesh.lamM=self.emphasis_on_marker_guess
        if self.record_gui:
            self.current_record.event_list.append(('emphasis_on_marker_guess',self.emphasis_on_marker_guess))
    @on_trait_change('smoothness')
    def update_smoo(self):
        self.current_mesh.lamL=self.smoothness
        if self.record_gui:
            self.current_record.event_list.append(('smoothness',self.smoothness))
#    @on_trait_change('radius_best_guess',emphasis_on_radius_guess,emphasis_on_marker_guess,smoothness')
#    def update_params(self):
#        self.current_mesh.r_est=self.radius_best_guess
#        self.current_mesh.lamR=self.emphasis_on_radius_guess
#        self.current_mesh.lamM=self.emphasis_on_marker_guess
#        self.current_mesh.lamL=self.smoothness
        
    def reset_params(self):
        self.radius_best_guess=self.current_mesh.r_est
        self.emphasis_on_radius_guess=self.current_mesh.lamR
        self.emphasis_on_marker_guess=self.current_mesh.lamM
        self.smoothness=self.current_mesh.lamL

    def read_from_marker_list(self,marker_list):
        labels=[str(i) for i in range(len(marker_list))]
        for label,marker in zip(labels,marker_list):
            self.segments[label]=dict(marker=np.array(marker))
    
    def read_from_markerfile(self,file_name):
        if file_name[:6]=='/work/':##file_name is already fullpath
            self.marker_file=file_name
        else:
            self.marker_file=self.env.read_dir+file_name
        f=open(self.marker_file,'r')
        file_extension=self.marker_file.split('.')[-1]
        raw_list=[[get_number(line),parse_line(line,file_extension)] for line in f if line[0]!='#']#in x, y, z
        for label,marker in raw_list:
            self.segments[label]=dict(marker=marker)
        f.close()


    def _time_changed(self):
        print 'time changed'
        self.worm.array_src.scalar_data=self.worm.daX[self.time].compute()
        unscaled_patch=self.worm.array_src.scalar_data
        self.cube.I=unscaled_patch
        self.update_cube()
        
        print 'time_chaged:unscaled_patch.mean():',unscaled_patch.mean()

        for neuron in self.segments.values():
            marker=neuron['marker'][::-1]
            window_shape=self.neuron_window_shape[::-1]
            if 'mesh' in neuron.keys():
                print 'mesh patch updated:',neuron
                unscaled_patch=get_Patch(marker,self.cube.I,window_shape)
                Iscale=unscaled_patch.max()
                p0d=unscaled_patch.astype(np.float)
                I=p0d/Iscale
                neuron['mesh'].I=I
                neuron['mesh'].Iscale=Iscale


#    def _to_var_fired(self):
#        X=self.worm.daX[:,
#                        self.cube.x_pos:self.cube.x_pos+self.cube.x_len+1,
#                        self.cube.y_pos:self.cube.y_pos+self.cube.y_len+1,
#                        self.cube.z_pos:self.cube.z_pos+self.cube.z_len+1].compute()
#        Xf=X.astype(np.float)/X.max()#12985 is Xmax
#        varX=np.sum(Xf*Xf,axis=0)
#        varX/=varX.max()
#        A=varX-varX.min()#set cube view
#        self.cube.array_src.scalar_data=A/A.max()*255
#        t=np.percentile(varX.ravel(),95)
#        boo_varX=(varX>t).astype(np.float)
#        self.current_neuron['mesh'].I=np.ascontiguousarray(boo_varX*20)

    def _do_global_iteration_fired(self):
        hold_neuron,hold_mesh=self.current_neuron,self.current_mesh
        for name in self.segments.keys():
            neuron=self.segments[name]
            if 'mesh' in neuron.keys():
                self.current_neuron=neuron
                self.current_mesh=neuron['mesh']
                if 'record' in neuron.keys():
                    self.current_record=neuron['record']
                    self._next_iteration_fired()
        self.current_neuron,self.current_mesh=hold_neuron,hold_mesh
            
    def _next_iteration_fired(self):
        print 'Volume_views3.py:_next_iteration_fired()'
        self.segments[self.neuron_name]['mesh'].update_plot()
        self.Average_Intensity=self.segments[self.neuron_name]['mesh'].interior_intensity
        if 'intensity' in self.current_neuron.keys():
            self.current_neuron['intensity'][str(self.time)]=self.Average_Intensity
        else:
            self.current_neuron['intensity']=dict({str(self.time):self.Average_Intensity})
        if self.record_gui:
            self.current_record.event_list.append('iteration')
            self.current_record.event_list.append('mesh state',self.current_mesh)
        
#        self.current_record.event_list.append( ('binary image',self.current_mesh.create_segmentation_image()) )
#        self.current_record.event_list.append( ('Radius',self.current_mesh.Radius) )


                
#        if hasattr(self,):#!!!!!!!
#        for 
#        self.current_mesh.I=self.worm.daX[self.time].compute()
    
#    def _down_fired(self): 
    #####Change Marker#####
    @on_trait_change('neuron_name') 
    def neuron_name_changed(self):
        name=self.neuron_name#.strip()
        print 'name is:',name
        try:
            self.m=self.segments.keys().index(name)
        except:
            print 'neuron name does not exist'
            return
#        self.update_region()#can assume called by _m_changed()
            
    @on_trait_change('m')
    def _m_changed(self):
        print ' '
        print 'mchanged fired'
#        mlab.clf(self.small_scene.mayavi_scene)
        self.time_region_start=time.time()
        self.clear_grids_from_figure()##Clears all grids from self.small_scene
        if self.m<0:self.m=0
        self.m_caused_name=True
        self.neuron_name=self.segments.items()[self.m][0]
        self.update_region()
        print 'time to update after neuron change ',time.time()-self.time_region_start



    def _reset_current_mesh_fired(self):
        print 'using incomplete method _reset !!! '
        self.clear_grids_from_figure()#Remove mesh from small scene
        self.current_mesh.whole_mesh.parent.parent.remove()
        del self.segments[self.neuron_name]['mesh']#delete mesh
        self.update_region()#try again

        
    def update_region(self):#Called exactly once per neuron change
#        print 'update_region fired'
#        self.update_box()
#        if not self.segments[self.neuron_name].has_key('mesh'):
#            self.segments[self.neuron_name]['is_segmented']=True ##A place holder right now. It means user is satisfied with this neuron segmentation
#            Patch=self.cube.unscaled_patch
##            Patch=self.cube.array_src.scalar_data
#            self.segments[self.neuron_name]['mesh']=SphereMesh(self.polar_cache,Patch,spacing=self.spacing)
#            offset=self.segments[self.neuron_name]['marker'][::-1] - self.window_radius[::-1]
#            self.segments[self.neuron_name]['mesh'].draw_mesh_on_whole_worm(self.whole_worm,offset)
#        
#        self.segments
#        self.current_neuron=self.segments[self.neuron_name]
#        self.current_mesh=self.segments[self.neuron_name]['mesh']##softcopy that should bind them.
#        self.reset_params()#set the traits on the GUI to reflect the active mesh
#        self.current_mesh.draw_mesh_on_small_scene(self.small_scene)
        print 'update_region fired'
        self.current_neuron=self.segments[self.neuron_name]
        self.update_box()
        if not self.current_neuron.has_key('mesh'):
            self.current_neuron['is_segmented']=True ##A place holder right now. It means user is satisfied with this neuron segmentation
            Patch=self.cube.unscaled_patch
            print 'update_region: unscaled_patch.mean():',Patch.mean()
#            Patch=self.cube.array_src.scalar_data
            self.current_neuron['mesh']=SphereMesh(self.polar_cache,Patch,spacing=self.spacing)
            offset=self.current_neuron['marker'][::-1] - self.window_radius[::-1]
            self.current_neuron['mesh'].draw_mesh_on_whole_worm(self.whole_worm,offset)
        if self.record_gui:
            if not self.current_neuron.has_key('record'):
                self.current_neuron['record']=GUIRecord(self.env,self.file_name)
            self.current_record=self.current_neuron['record']
        self.current_mesh=self.segments[self.neuron_name]['mesh']##softcopy that should bind them.
        self.reset_params()#set the traits on the GUI to reflect the active mesh
        self.current_mesh.draw_mesh_on_small_scene(self.small_scene)

        try:
            self.marker_status=self.current_neuron['notes']['quality']
        except:
            self.marker_status=self.marker_status_options[0]

        self.meshes_in_view=list_meshes_in_cube(self.cube,self.neuron_name,self.segments)
        self.surrounding_mesh_list=[]
        for key,offset in self.meshes_in_view:
            self.segments[key]['mesh'].draw_mesh_on_scene_with_offset_without_update(self.small_scene,offset)
            self.surrounding_mesh_list.append(self.segments[key]['mesh'])
        self.set_mesh_visibility()

###########Saving Methods ###########
    def save_segments(self,global_offset=np.array([0,0,0])):#use global_offset if tiff image is a part of a larger image.#x,y,z
        if not self.save_at_end:
            return
        print 'save_segments()'
        def fun(key):
            try:
                return self.segments[key]['is_segmented']
            except:
                return False
                
        filtrate=filter(lambda x:fun(x[-1]), enumerate(self.segments.keys()))
        for m,name in filtrate:
            #offset is z,y,x to match up with images...sorry its confusing...
            offset=np.array( self.segments[name]['marker'][::-1] ) - self.window_radius[::-1] + global_offset[::-1]
            bool_array=self.segments[name]['mesh'].create_segmentation_image()#Inside is 1, outside is 0
            float_array=self.segments[name]['mesh'].I * self.segments[name]['mesh'].Iscale
            try:
                notes=self.segments[name]['notes']
            except:
                notes=None
            segment=Segment(name,m,offset,bool_array,float_array,notes)#Class I use for Saving data
        
            f_name=self.env.write_dir + 'Segment_'+ name + '.pkl'
        
            with open(f_name,'wb') as handle:
                dill.dump(segment,handle)

############## Leaf Methods  ######### (they don't call other things)     ################
    def _show_right_panel_fired(self):
        print 'show right panel'
        print self.display_right_panel
        self.display_right_panel=not self.display_right_panel
        print self.display_right_panel

    def _show_global_view_fired(self):
        print 'show global left panel'
        print self.display_global_view
        self.display_global_view=not self.display_global_view
        print self.display_global_view
        
    def _bad_marker_fired(self):##Not implemented correctly yet
#        self.segments[self.neuron_name]['is_segmented']=False
        self.segments[self.neuron_name]['notes']=dict(quality='marker is bad')
        self.marker_status=self.marker_status_options[1]
        
    def update_box(self):
        print 'update_box'
#        mark=self.segments[self.neuron_name]['marker']
        mark=self.current_neuron['marker']
        temp_mark=np.array([mark[2],mark[1],mark[0]])#temp_mark is zyx
        temp_window_radius=np.array([self.window_radius[2],self.window_radius[1],self.window_radius[0]])#zyx
        
        self.cube.x_pos=temp_mark[0]-temp_window_radius[0]
        self.cube.y_pos=temp_mark[1]-temp_window_radius[1]
        self.cube.z_pos=temp_mark[2]-temp_window_radius[2]
        self.cube.x_len=2*temp_window_radius[0]+1
        self.cube.y_len=2*temp_window_radius[1]+1
        self.cube.z_len=2*temp_window_radius[2]+1
        self.cube.dirty=True

    @on_trait_change('box_x_pos,box_y_pos,box_z_pos,box_x_len,box_y_len,box_z_len')##causes many unnecssary redundant updates
    def update_cube(self):
        self.cube.dirty=True
#        self.cube.update()#Redunant because is already called on trait change



    def _down_fired(self):
        self.multivolume.factor+=1
        self.multivolume.update()
    def _up_fired(self):
        self.multivolume.factor-=1
        self.multivolume.update()
#        c1=self.cube.array_src.children[0]
#        lut=c1.scalar_lut_manager.lut
#        lut.table=debug_lut_array[0]

    def _debug_fired(self):
        print ''
        print 'debug fired'
#        print 'nchildren is ',len(self.cube.array_src.children)
#        c1=self.cube.array_src.children[0]
#        lut=c1.scalar_lut_manager.lut
#        print 'table range:',lut.table_range
#        print 'lut table:'
#        print lut.table.to_array()        
#        
#        debug_lut_array.append(lut.table.to_array())
#        A=np.copy(self.cube.view_Image)
        print 'cube scalar_data max:',self.cube.array_src.scalar_data.max()       
        self.cube.array_src.scalar_data*=2
#        print 'cube. view_Image max:',self.cube.view_Image.max()
        print ' '
    
        
    @on_trait_change('scene.activated')
    def scene_activated(self):
        print 'scene is activated'
#        self.scene.scene.interactor.interactor_style = \
#            tvtk.InteractorStyleImage()#no 3d rotation
#            tvtk.InteractorStyleTerrain()
        iren=self.scene.scene.interactor
        iren.add_observer("RightButtonPressEvent",ButtonEvent)
        iren.add_observer("KeyPressEvent",cbk)
        print 'changed style'

    def _toggle_mesh_visibility_fired(self):
        print 'toggle mesh fired'
        self.mesh_visibility=not self.mesh_visibility
        self.set_mesh_visibility()
    def set_mesh_visibility(self):
        self.current_mesh.small_mesh.actor.actor.visibility= self.mesh_visibility
        for mesh in self.surrounding_mesh_list:
            print 'this loop works'
            mesh.surrounding_mesh.actor.actor.visibility= self.mesh_visibility
            

    def euclid_length(self,vec1):
        vec1=np.array(vec1)
        vec1*=self.spacing
        return np.linalg.norm(vec1)

    def _previous_marker_fired(self):
        if self.m is 0:
            print 'already at first marker'
            return
        self.m-=1
    def _next_marker_fired(self):
        if self.m is len(self.segments.items())-1:
            print 'already at last marker'
            return
        self.m+=1
    
    def _forward_fired(self):
        if self.time==self.max_time:
            print 'already at last timepoint'
        else:
            self.time+=1
    def _backward_fired(self):
        if self.time==self.min_time:
            print 'already at first timepoint'
        else:
            self.time-=1

    def _save_session_fired(self):
        self.save_at_end=not self.save_at_end
        if self.save_at_end:
            self.save_status=self.save_status_options[1]
        elif not self.save_at_end:
            self.save_status=self.save_status_options[0]

#    @on_trait_change('m')#try call directly instead
    def clear_grids_from_figure(self):
        if hasattr(self,'current_mesh'):
            print 'clear_grids_from_figure() fires'
            MScene=self.small_scene.mayavi_scene
            grid_children=[]
            for child in MScene.children:
                if child.name is 'GridSource':
                    grid_children.append(child)
            for child in grid_children:
                MScene.remove_child(child)


        



#'''
    dashboard= VGroup(
                        'Average_Intensity',
                        HGroup(Item('next_iteration',show_label=False),Item('do_global_iteration',show_label=False)),
                        'radius_best_guess',
                        'emphasis_on_radius_guess',
                        'emphasis_on_marker_guess',
                        'smoothness',
                    )
#    ui_box_x_pos=Item('box_x_pos',editor=RangeEditor(auto_set=False,enter_set=True,mode='slider'))        
#    ui_box_y_pos=Item('box_y_pos',editor=RangeEditor(auto_set=False,enter_set=True,mode='slider'))
##    ui_box_z_pos='box_z_pos'
#    ui_box_z_pos=Item('box_z_pos',editor=RangeEditor(auto_set=False,enter_set=True,mode='slider'))         
#    ui_box_x_len=Item('box_x_len',editor=RangeEditor(auto_set=False,enter_set=True,mode='slider')) 
#    ui_box_y_len=Item('box_y_len',editor=RangeEditor(auto_set=False,enter_set=True,mode='slider'))
#    ui_box_z_len=Item('box_z_len',editor=RangeEditor(auto_set=False,enter_set=True,mode='slider'))
    
#    UI_Box=VGroup(HGroup(ui_box_x_len,ui_box_y_len,ui_box_z_len),HGroup(ui_box_x_pos,ui_box_y_pos,ui_box_z_pos))

    UI_Box=VGroup(HGroup('box_x_len','box_y_len','box_z_len'),HGroup('box_x_pos','box_y_pos','box_z_pos'))
    
#    def _init_is_finished_changed(self):
#    def default_traits_view(self):

    traits_view = View( 
#       self.add_trait('traits_view', View( 
            HGroup(
                        VGroup(Item('whole_worm',editor=SceneEditor(scene_class=MayaviScene),height=250, width=300, show_label=False),
#                                    HGroup('_cube.x_len','_cube.y_len','_cube.z_len'),                          
#                                    HGroup('_cube.x_pos','_cube.y_pos','_cube.z_pos'),
#                                    HGroup('cube.x_len','cube.y_len','cube.z_len'),                          
#                                    HGroup('cube.x_pos','cube.y_pos','cube.z_pos'),
    #                               Include('cube_display')
                                   UI_Box,
#                                   HGroup('box_x_pos',Item('box_y_pos',editor=RangeEditor(low=1,high=100,auto_set=False,enter_set=True,mode='slider')),'box_z_pos'),
#                                    HGroup('box_x_len','box_y_len','box_z_len'),
                                     
                                    HGroup('down','up'),
#                                    spring
#                                    resizable=True,#fails
                                    visible_when='display_global_view==True'
                                    ),
    
            VGroup(
                        Item('small_scene',editor=SceneEditor(scene_class=MayaviScene),height=250, width=300, show_label=False),
                            VGroup(#Include('main_scene'),
                                HGroup(spring,'show_right_panel'),
                                HGroup( Item('neuron_name',editor=TextEditor(auto_set=False,enter_set=True,evaluate=str)) ,'previous_marker','next_marker'),
                                HGroup('backward',Item('time'),'forward'),
#                                HGroup(Item('to_var',show_label=False),Item('to_normal',show_label=False)),
                                Include('dashboard'),
                            HGroup('save_session',Item('save_status',style='readonly',show_label=False)),
                            HGroup('bad_marker',Item('marker_status',style='readonly',show_label=False)),
                                HGroup('toggle_mesh_visibility','reset_current_mesh','debug')
                                ),
#                                resizeable=True,#fails
                   ),
            VGroup(
    ##                    Item('cube.vol_module',style='custom',height=250, width=300, show_label=False,enabled_when='cube.vol_module.activated')
    ##                    Item('cube.vol_module',style='custom',height=250, width=300, show_label=False)
                        Item('volume_control',style='custom',height=250, width=100, show_label=False),
#                        spring,
                           visible_when='display_right_panel==True',     
                    )
                ),
    #            editor=SceneEditor(scene_class=Scene),
    #                ,
    #            VGroup(
    #                    
    #                ),
                resizable=True,
                
                )
                
#'''     
                
#        return traits_view
        
#        )
#        self.add_trait('traits_view',traits_view)
        

if __name__=='__main__':

    #path='/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/150226 Hi-res_40x Full Worm W5.tif'#Real big
    #path='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/150303_ZIM294_L4_W2 Full Worm.tif'
    #directory='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/'

#    data='150303'
#    data='150731'
  
    data='150311_20x_W1-1'
#    data='150311_20x_W1-2'

#    data='150311_40x_W1-1'
#    data='150311_40x_W1-2'
#    data='150311_40x_W1-3'

    if data is '150303':
        path='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/150303_ZIM294_L4_W2 Full Worm.tif'
        read_dir='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/'
        write_dir='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/Segmentations/1/'
        marker_file='Segmentations/1/150303_CM_Hi-res_20x annotation chris on 150618.apo'    
    elif data is '150731':
        path='/work/03176/csnyder/Volumes/150731_CM_Device_II_Stimulus/W1/c_150731_Glycerol_1M_W1_O20X_F630x150um_P11008x1000S210_m2_resize_stack.tif'
        read_dir='/work/03176/csnyder/Volumes/150731_CM_Device_II_Stimulus/W1/'
        write_dir='/work/03176/csnyder/Volumes/150731_CM_Device_II_Stimulus/W1/'
        marker_file='c_150731_Glycerol_1M_W1_O20X_F630x150um_P11008x1000S210_m2_resize_stack_20150805SM.apo'


    elif data is '150311_20x_W1-1':
        path='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/20x/W1/stacks/150311_ZIM294_L4_W1-1_z60um_O20X_F630x150um_P11008x1000S178_resized_RAW_stack.tif'
        read_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/20x/W1/'
        write_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/20x/W1/Segments/W1-1/'
        marker_file='sudip piecewise annotations/150311_ZIM294_L4_W1-1_20150804SM_nonredundant.apo'
        global_offset=np.array([0,60,20])
    elif data is '150311_20x_W1-2':
        path='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/20x/W1/stacks/150311_ZIM294_L4_W1-2_z60um_O20X_F630x150um_P11008x1000S178_resized_RAW_stack.tif'
        read_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/20x/W1/'
        write_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/20x/W1/Segments/W1-2/'
        marker_file='sudip piecewise annotations/150311_ZIM294_L4_W1-2_20150804SM-nonredundant.apo'
        global_offset=np.array([1719,0,0])



    elif data is '150311_40x_W1-1':
        path='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/stacks/150311_ZIM294_L4_W1-1_z42um_N40X_F350x150um_P11008x1000S174_resized_RAW_stack.tif'
        read_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/'
        write_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/Segments/W1-1/'
        marker_file='sudip piecewise annotations/150311_ZIM294_L4_W1-1_20150804SM-nonredundant.apo'
        global_offset=np.array([0,107,10])
        
    elif data is '150311_40x_W1-2':
        path='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/stacks/150311_ZIM294_L4_W1-2_z42um_N40X_F350x150um_P11008x1000S174_resized_RAW_stack.tif'
        read_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/'
        write_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/Segments/W1-2/'
        marker_file='sudip piecewise annotations/150311_ZIM294_L4_W1-2_SM20150731-nonredundant.apo'
        global_offset=np.array([1884,80,8])
        
    elif data is '150311_40x_W1-3':
        path='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/stacks/150311_ZIM294_L4_W1-3_z42um_N40X_F350x150um_P11008x1000S174_resized_RAW_stack.tif'
        read_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/'
        write_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/Segments/W1-3/'
        marker_file='sudip piecewise annotations/150311_ZIM294_L4_W1-3_SM20150731-nonredundant.apo'
        global_offset=np.array([4074,0,0])




#'/work/03176/csnyder/Volumes/150731_CM_Device_II_Stimulus/W1/c_150731_Glycerol_1M_W1_O20X_F630x150um_P11008x1000S210_m2_resize_stack_20150805SM.apo'

    
    spacing=np.array([2.0,1.0,1.0])
#    spacing=np.array([1.0,1.0,1.0])
    
    
    #I=io.imread(patchpath)
    I0=io.imread(path)
    print 'finished reading data'
    
    
    env=Environment(read_dir,write_dir)
    
    vis=Vis(wholeworm=I0,env=env,spacing=spacing)
    vis.read_from_markerfile(marker_file)
    
    vis.configure_traits()
    
    
    try:
        vis.save_segments(global_offset)
    except:
        vis.save_segments()


    
    
    #import cPickle as pickle#python2
    ###Pickle Class for later
    
#    filename=write_dir+'first_orddict_pickle.p'
    #with open(filename,'wb') as handle:
    #    pickle.dump(vis.segments,handle)
    #
    #SaveObject():
        
#    class C:
#        pass
#    c=C()
#    
#    import dill
#    #    dill.dump(vis,handle)
#    
#    A=vis.segments['2']['mesh']
#    with open(filename,'w') as handle:
#        dill.dump(A,handle)
#    
#    with open(filename+'c','w') as handle:
#        dill.dump(c,handle)
#    
#    dill.dump_session('session.dill')
#    
#    
#    with open(filename,'wb') as handle:
#        dill.dump_session(handle)
#    
#    with open(write_dir+'segments.dill','wb') as handle:
#        dill.dump(vis.segments,handle)
#    
    #vis=Vis(I,wholeworm=I)
    ####M=MultiResolution(I0)
    #print 'finished reading'
    #
    #M=MultiArray(I0)
    #J=M.downsample_by(10)
    #
    #print 'done downsampling'
    #
    ##i = tvtk.ImageData(dimensions=(74,74,34),spacing=spacing, origin=(0, 0, 0))
    ###i = tvtk.ImageData(dimensions=I.shape,spacing=(1, 1, 1), origin=(0, 0, 0))
    ####i = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
    ####i.point_data.scalars = np.ones(I.shape).ravel()#I.ravel()
    #
    ##i = tvtk.ImageData(dimensions=np.roll(J.shape,1),spacing=(1, 1, 1), origin=(0, 0, 0))
    #i = tvtk.ImageData(dimensions=J.shape[::-1],spacing=(1, 1, 2), origin=(0, 0, 0))
    #i.point_data.scalars = J.ravel()
    #i.point_data.scalars.name = 'I_scalars'
    #
    #mlab.pipeline.volume(i)
    
    
    #i = tvtk.ImageData(dimensions=(74,74,34),spacing=(1, 1, 1), origin=(0, 0, 0))
    ##i = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
    ##i.point_data.scalars = np.ones(I.shape).ravel()#I.ravel()
    #i.point_data.scalars = I.ravel()
    #
    ##i.point_data.scalars.name = 'I_scalars'
    ##i.dimensions = I.shape
    #
    
    
    
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
    
    
