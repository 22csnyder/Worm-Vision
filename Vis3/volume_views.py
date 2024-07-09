# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 09:38:41 2015

@author: csnyder
"""
import skimage.io as io

from mayavi import mlab
from scipy.ndimage import zoom


import numpy as np
from traits.api import HasTraits,on_trait_change, Instance,Button,Float,DelegatesTo,Str,Range,Int
from traitsui.api import View,VGroup,HGroup,Include,TextEditor,Item
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from tvtk.pyface.scene_editor import SceneEditor
from tvtk.api import tvtk

import time
from numpy.lib import stride_tricks
from mayavi.core.api import Engine
from mayavi.sources.api import ArraySource
from mayavi.modules.volume import Volume

from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.modules.structured_grid_outline import StructuredGridOutline
from traits.api import Interface

from save_patches import parse_line,get_number
from multi_obj_polar_nucleus3D import PolarCache
from WormBox.BaseClasses import Environment#,Nucleus
from WormBox.WormSources import MultiVolume,CubeMesh,SphereMesh

import collections
import dill

debug_lut_array=[]

class Segment:###Don't Change this class or risk losing the .pkl
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
        

def cbk(vtk_obj,event):
    print vtk_obj.GetClassName(),event,vtk_obj.GetKeyCode()
    #use tvtk.to_tvtk(vtk_obj) to get the tvtk version of this object
    
def ButtonEvent(obj,event):
    if event== "RightButtonPressEvent":
        print 'right button was pressed'
	        


#from Do_sphharm_segmentations import SlideShow
class Vis(HasTraits):
    button=Button('next_iteration')
    reset_current_mesh=Button('retry')

    bad_marker=Button('bad_marker')    
    
    Average_Intensity=Float()
    
    
    radius_best_guess=Range(low=1.0,high=15.0,)        
    emphasis_on_radius_guess=Range(low=0.0,high=30.0)
    smoothness=Range(low=0.0,high=500.0)     
    emphasis_on_marker_guess=Range(low=0.0,high=500.0)    
    
    toggle_mesh_visibility=Button('on/off')
    
    neuron_name=Str('')
    m=Int(-1)
    next_marker=Button()
    previous_marker=Button()
#    save_segmentation=Button() ##Button as of yet unimplemented
    
    down=Button('sample') 
    up=Button('sample')

    debug=Button('debug')    
    
    engine1 = Instance(Engine, args=())
    engine2 = Instance(Engine, args=())
    whole_worm      = Instance(MlabSceneModel,())
    small_scene           = Instance(MlabSceneModel,())
#    integral=Float(1.2)


    cube=Instance(CubeMesh(),())
#    cube=Instance(CubeMesh,())#Either seem to work

    box_x_len=DelegatesTo(delegate='cube',prefix='x_len')
    box_y_len=DelegatesTo(delegate='cube',prefix='y_len')
    box_z_len=DelegatesTo(delegate='cube',prefix='z_len')
    box_x_pos=DelegatesTo(delegate='cube',prefix='x_pos')
    box_y_pos=DelegatesTo(delegate='cube',prefix='y_pos')
    box_z_pos=DelegatesTo(delegate='cube',prefix='z_pos')
    
#    cube_display=VGroup('box_coords.x_len','box_coords.y_len','box_coords.z_len','box_coords.x_pos','box_coords.y_pos','box_coords.z_pos')

    def _whole_worm_default(self):
        self.engine1.start()
        return MlabSceneModel(engine=self.engine1)

    def _small_scene_default(self):
        self.engine2.start()
        return MlabSceneModel(engine=self.engine2)
        
    def __init__(self,wholeworm,env,spacing):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        self.env=env
        self.spacing=spacing
        
        self.I=wholeworm
        self.multivolume=MultiVolume(self.I,init_downsample=10,spacing=self.spacing)#its' a big volume
        self.multivolume.display_on_scene( self.whole_worm )
        
#        s=mlab.pipeline.scalar_field(I,figure=self.scene.mayavi_scene)
#        self.volume=mlab.pipeline.volume(s,figure=self.scene.mayavi_scene)

        self.cube.spacing=self.spacing
#        self.cube=CubeMesh()
        self.cube.observe_data(self.multivolume.multiImage.orig_Array)
        self.cube.display_grid_on_scene(self.whole_worm)
        self.cube.display_volume_data_on_scene(self.small_scene)

        self.segments=collections.OrderedDict()
        
#        self.add_trait('cube_param',self.cube.return_gui())        
        
#        self.add_trait('box_x_pos',DelegatesTo('cube',prefix='x_pos'))
#        self.add_trait('box_y_pos',DelegatesTo('cube',prefix='y_pos'))
#        self.add_trait('box_z_pos',DelegatesTo('cube',prefix='z_pos'))
        
        neuron_window=np.array([75,75,35])#must be odd #Falsh
        self.window_radius=np.array((neuron_window-1)//2)#x,y,z        
        self.patch_shape=2*self.window_radius[::-1]
        self.polar_cache=PolarCache(self.patch_shape,self.spacing)
        print 'vis is init'

    @on_trait_change('radius_best_guess')
    def update_rad(self):
        self.current_mesh.r_est=self.radius_best_guess
    @on_trait_change('emphasis_on_radius_guess')
    def update_lamR(self):
        self.current_mesh.lamR=self.emphasis_on_radius_guess
    @on_trait_change('emphasis_on_marker_guess')
    def update_lamM(self):
        self.current_mesh.lamM=self.emphasis_on_marker_guess
    @on_trait_change('smoothness')
    def update_smoo(self):
        self.current_mesh.lamL=self.smoothness
    
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


    def read_from_markerfile(self,file_name):
        self.marker_file=self.env.read_dir+file_name
        f=open(self.marker_file,'r')
        file_extension=self.marker_file.split('.')[-1]
        raw_list=[[get_number(line),parse_line(line,file_extension)] for line in f if line[0]!='#']#in x, y, z
        for label,marker in raw_list:
            self.segments[label]=dict(marker=marker)
        f.close()

    
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
        if self.m<0:self.m=0
        self.m_caused_name=True
        self.neuron_name=self.segments.items()[self.m][0]
        self.update_region()
        print 'time to update after neuron change ',time.time()-self.time_region_start



    ####WARNING: Currently this method does not clear mesh from the big figure#####
    def _reset_current_mesh_fired(self):
        print 'using incomplete method _reset !!! '
        self.clear_grids_from_figure()#Remove mesh from small scene
        del self.segments[self.neuron_name]['mesh']#delete mesh
        self.update_region()#try again


        
    def update_region(self):#Called exactly once per neuron change
        print 'update_region fired'
        self.update_box()
        if not self.segments[self.neuron_name].has_key('mesh'):
#            Patch=np.copy(self.cube.view_Image)
            self.segments[self.neuron_name]['is_segmented']=True ##A place holder right now. It means user is satisfied with this neuron segmentation
#            Patch=self.cube.view_Image
            Patch=self.cube.array_src.scalar_data
            self.segments[self.neuron_name]['mesh']=SphereMesh(self.polar_cache,Patch,spacing=self.spacing)
            offset=self.segments[self.neuron_name]['marker'][::-1] - self.window_radius[::-1]
            self.segments[self.neuron_name]['mesh'].draw_mesh_on_whole_worm(self.whole_worm,offset)
        self.current_mesh=self.segments[self.neuron_name]['mesh']##softcopy that should bind them.
        self.reset_params()#set the traits on the GUI to reflect the active mesh
        self.current_mesh.draw_mesh_on_small_scene(self.small_scene)

        ###get list of all meshes in the area #and plot them
        if hasattr(self,'current_mesh'):
            print 'has attr current_mesh'
            center_main=np.array(self.segments[self.neuron_name]['marker'][::-1])
#            radius_main=self.current_mesh.r0
#            radius_main=self.segments[self.neuron_name]['mesh'].r0
            self.meshes_in_view=[]
            for key in self.segments.keys():
                if self.segments[key].has_key('mesh') and key is not self.neuron_name :#initialized meshes besides the current one
                    center_other=np.array(self.segments[key]['marker'][::-1])
                    xl=self.box_x_pos<center_other[0]<self.box_x_pos+self.box_x_len
                    yl=self.box_y_pos<center_other[1]<self.box_y_pos+self.box_y_len
                    zl=self.box_z_pos<center_other[2]<self.box_z_pos+self.box_z_len
                    if xl and yl and zl:
                        offset=center_other-center_main
                        self.meshes_in_view.append([key,offset])
                        
#                    radius=self.segments[key]['mesh'].r0
#                    if self.euclid_length(offset)<radius+radius_main:
#                        self.meshes_in_view.append([key,offset])
#            key_in_range=[]
#            for key in self.segments.keys():
#                if self.segments[key].has_key('mesh') and key is not self.neuron_name :#initialized meshes besides the current one
#                    center_other=np.array(self.segments[key]['marker'][::-1])
#                    radius=self.segments[key]['mesh'].r0
#                    offset=center_other-center_main
#                    if self.euclid_length(offset)<radius+radius_main:
#                        key_in_range.append([key,offset])
            
            for key,offset in self.meshes_in_view:
                self.segments[key]['mesh'].draw_mesh_on_scene_with_offset_without_update(self.small_scene,offset)

#    @on_trait_change('box_y_pos')#Interesting experiment
#    @on_trait_change('box_z_pos')
#    @on_trait_change('box_x_len')
#    @on_trait_change('box_y_len')
#    @on_trait_change('box_z_len')
    
#    @on_trait_change('box_x_pos')
#    @on_trait_change('cube.xlen')
    @on_trait_change('box_x_pos,box_y_pos,box_z_pos,box_x_len,box_y_len,box_z_len')##causes many unnecssary redundant updates
    def update_cube(self):
        print 'updating cube'
        self.cube.update()

###########Saving Methods ###########
    def save_segments(self,global_offset=np.array([0,0,0])):#use global_offset if tiff image is a part of a larger image.#x,y,z
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
            segment=Segment(name,m,offset,bool_array,float_array)#Class I use for Saving data
        
            f_name=self.env.write_dir + 'Segment_'+ name + '.pkl'
        
            with open(f_name,'wb') as handle:
                dill.dump(segment,handle)

############## Leaf Methods  ######### (they don't call other things)     ################
    def bad_marker_fired(self):
        self.segments[self.neuron_name]['is_segmented']=False
        self.segments[self.neuron_name]['notes']=dict(quality='marker is bad')
        
        
    def update_box(self):
        print 'update_box'
        mark=self.segments[self.neuron_name]['marker']
        temp_mark=np.array([mark[2],mark[1],mark[0]])#Just depends on the format of image X
        temp_window_radius=np.array([self.window_radius[2],self.window_radius[1],self.window_radius[0]])
        
        self.box_x_pos=temp_mark[0]-temp_window_radius[0]
        self.box_y_pos=temp_mark[1]-temp_window_radius[1]
        self.box_z_pos=temp_mark[2]-temp_window_radius[2]
        self.box_x_len=2*temp_window_radius[0]
        self.box_y_len=2*temp_window_radius[1]
        self.box_z_len=2*temp_window_radius[2]

    @on_trait_change('button')
    def update_current_mesh(self):
        print 'update mesh fired'
        self.segments[self.neuron_name]['mesh'].update_plot()
        self.Average_Intensity=self.segments[self.neuron_name]['mesh'].interior_intensity

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
        if hasattr(self,'mesh'):
            self.current_mesh.small_mesh.actor.actor.visibility= not self.current_mesh.small_mesh.actor.actor.visibility

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

    @on_trait_change('m')
    def clear_grids_from_figure(self):
        if hasattr(self,'current_mesh'):
            MScene=self.small_scene.mayavi_scene
            grid_children=[]
            for child in MScene.children:
                if child.name is 'GridSource':
                    grid_children.append(child)
            for child in grid_children:
                MScene.remove_child(child)

    dashboard= VGroup(
                        'Average_Intensity',
                        'button',
                        'radius_best_guess',
                        'emphasis_on_radius_guess',
                        'emphasis_on_marker_guess',
                        'smoothness',
                    )
    traits_view = View( HGroup(
                    VGroup(Item('whole_worm',editor=SceneEditor(scene_class=MayaviScene),height=250, width=300, show_label=False),
#                            HGroup('cube.x_len','cube.y_len','cube.z_len')                           
#                            HGroup('x_pos','y_pos','z_pos')   
#                               Include('cube_display')
                               HGroup('box_x_pos','box_y_pos','box_z_pos'),
                                HGroup('box_x_len','box_y_len','box_z_len'),
                                HGroup('down','up')

                           ),
                VGroup(
                    Item('small_scene',editor=SceneEditor(scene_class=MayaviScene),height=250, width=300, show_label=False),
                        VGroup(#Include('main_scene'),
                            HGroup( Item('neuron_name',editor=TextEditor(auto_set=False,enter_set=True,evaluate=str)) ,'previous_marker','next_marker'),
                            Include('dashboard'),#'save_segmentation',
                            HGroup('toggle_mesh_visibility','reset_current_mesh','bad_marker','debug')
                            )
                    
                    
                    )
                    )
#            editor=SceneEditor(scene_class=Scene),
#                ,
#            VGroup(
#                    
#                ),
                )

if __name__=='__main__':

    #path='/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/150226 Hi-res_40x Full Worm W5.tif'#Real big
    #path='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/150303_ZIM294_L4_W2 Full Worm.tif'
    #directory='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/'

#    data='150303'
    data='150731'
  
#    data='150311_20x_W1-1'
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
    
    
