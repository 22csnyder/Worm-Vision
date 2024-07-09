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
from traitsui.api import RangeEditor,Group

from traits.api import HasTraits,on_trait_change,Instance,Button,Float,DelegatesTo,Str,Range,Int,Bool
from mayavi.core.api import Engine
from mayavi.tools.mlab_scene_model import MlabSceneModel


#from Interactor import Interactor
from WormBox.VisClasses import Channel,MultiChannelWorm

from BoxCoords import BoxCoords
from traitsui.api import View,VGroup,HGroup,Include,TextEditor,Item,spring,HSplit,VSplit,VGrid
from mayavi.core.ui.mayavi_scene import MayaviScene
from tvtk.pyface.scene_editor import SceneEditor
#from tvtk.pyface.list_editor import ListEditor
#from tvtk.pyface.list_editor import ListEditor
from traitsui.editors import ListEditor

from traits.api import HasTraits

from collections import OrderedDict

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

from WormScene import WormScene

from NeuronAnalyzer import NeuronAnalyzer
from NeuronTrace import TracePlot
from MotionCorrection import MotionCorrection

#Maybe move to PyQt in the future

class Display(HasTraits):
    

    
#    max_time=Int(1)
#    min_time=Int(0)
#    time=Range('min_time','max_time',mode='slider')
#    
#    do_global_iteration=Button()
    
    reset_current_mesh=Button('retry')
#    

    next_iteration=Button()#do iteration
    Average_Intensity=Float()
    radius_best_guess=Range(low=1.0,high=15.0,)        
    emphasis_on_radius_guess=Range(low=0.0,high=60.0)
    smoothness=Range(low=0.0,high=500.0)     
    emphasis_on_marker_guess=Range(low=0.0,high=500.0)
    step_size=Range(0.001,0.1,)    
    
    
    volume_control=Instance(Volume,())
    

    toggle_mesh_visibility=Button('mesh on/off')
    def _toggle_mesh_visibility_fired(self):
        print 'toggle mesh fired'
        self.mesh_visibility=not self.mesh_visibility
        self.set_mesh_visibility()
    def set_mesh_visibility(self):
        try:
            self.current_mesh.small_mesh.actor.actor.visibility= self.mesh_visibility
        except:
            print 'no mesh found to toggle visibility'
            
#        for mesh in self.surrounding_mesh_list:
#            print 'this loop works'
#            mesh.surrounding_mesh.actor.actor.visibility= self.mesh_visibility
            
#    neuron_name=Str('')
#    m=Int(-1,record=True)
##    n_ind=Int(-1)
#    

#    save_session=Button() ##Button as of yet unimplemented
#    save_at_end=Bool(False)
    
    save_name=Str('')
    save_segments=Button()
    def _save_segments_fired(self):
        self.worm.save_segments(self.save_name)
        
        self.analyzer.save_intensity()
        
        
#    down=Button('sample')
#    up=Button('sample')
#
#    forward=Button('+')
#    backward=Button('-')
#
#    debug=Button('debug')    
#    
    debug_string=Str('')
#    whole_worm      = Instance(MlabSceneModel,())
#    small_scene     = Instance(MlabSceneModel,())
#    
#    mesh_visibility=Bool(True)
#    
#    current_sphere_mesh= Instance(SphereMesh)
 
    
#    cube_display=VGroup('box_coords.x_len','box_coords.y_len','box_coords.z_len','box_coords.x_pos','box_coords.y_pos','box_coords.z_pos')
    
#    bad_marker=Button('mark as bad')
#    marker_status_options=['','marker is bad']
#    marker_status=Enum(marker_status_options)
#    
#    init_is_finished=Bool(False)
#    save_status_options=['session will not save at end','session will save at end']
#    save_status=Enum('session will not save at end',['session will not save at end','session will save at end'])
#    
    

        
    
        
    show_right_panel=Button('more');display_right_panel=Bool(False)
    def _show_right_panel_fired(self):
        self.display_right_panel=not self.display_right_panel
        
        
        
#    scene_ = Instance(WormScene,())#wildcard causes problems with scene_dict
#_____________
    worm=Instance(MultiChannelWorm,())
    channels=Instance(Channel,())    
    
    
    hash_volume=Instance(dict)

#    def _large_scene_default(self):
#        self.engine1.start()
#        s=WormScene(engine=self.engine1)
#        s.name='large scene'
#        s.downsample=6
#        return s
#
#    def _small_scene_default(self):
#        self.engine2.start()
#        s=WormScene(engine=self.engine2)
#        s.name='small scene'
#        return s

    analyzer=Instance(NeuronAnalyzer,())

    def start(self):#wrapper 
        self.configure_traits()

#    sync=Button()##BLEH failed camera sync attempt
#    def _sync_fired(self):
#        self.sc1=self.scene_dict['local_cm1']
#        self.sc2=self.scene_dict['local_cm2']        
#        self.sc1.scene.sync_trait('camera',self.sc2.scene,mutual=True)
        

    def __init__(self,window_shape=None,spacing=None,smear=None,*worm_args):
        HasTraits.__init__(self)
        
        
#        self.trace=BoxCoords([3,3,3],[3,3,3])#DEBUG        
        
        
        
        self.scene_dict=OrderedDict()
        
        
#        self.worm=Channel(self.Scenes,window_shape,spacing,smear,*worm_args)#also plots the worms
#        self.worm=Channel(window_shape,spacing,smear,*worm_args)#also plots the worms
#        self.worm.assign_Scenes(self.Scenes)

#        self.sync_trait('volume_control',self.worm,'vol_module',mutual=True)
        
#        if window_shape is None:
#            self.window_shape=np.array(self.worm.X.shape)
#        else:
#            self.window_shape=np.array(window_shape)
        
        #These didnt' work for some reason
#        self.on_trait_change(self.draw_volumes,'hash_volume',remove=True)#Only do this when it's set at the start
#        self.on_trait_change(self.draw_volumes,'hash_volume')#Only do this when it's set at the start

#    def update_tf(self):
#        print 'was called'


    def draw_text(self):
        for key,neuron in self.worm.segments.items():
            p=neuron['marker']//self.worm['cm'].nm_voxel_shape
            mlab.text(x=p[0],y=p[1],z=p[2],text=key,orrient_to_camera=True,figure=self.scene_dict.values()[0].mayavi_scene)



    show_trace=Bool(False)
    @on_trait_change('hash_volume')
    def draw_volumes(self):
        
#        if hasattr(self.worm.segments,'seg_channel'):
#            self.sync_color_control()
    
#        self.on_trait_change(self.update_tf,'volume_control._volume_property')
        print 'hash_volume changed'
        for scene_name,channel_name in self.hash_volume.items():
            print 'channel:',channel_name,'scene:',scene_name
            channel=self.worm.channels[channel_name]
#            self.sync_trait('volume_control',channel,'small_vol_module',mutual=True)
            channel.draw_on_scene(self.scene_dict[scene_name])
#            channel.draw_on_scene(self.scene_dict[scene_name],self.volume_control)
        self.check_for_cube()


        if hasattr(self.worm,'time_series_data'):
            self.show_trace=True
            self.analyzer=NeuronAnalyzer()
            self.analyzer._init_from_gui(self.worm.segments,self.worm.time_series_data,self.worm.results_dir)
#            self.mc=MotionCorrection(self.worm.time_series_data,self.worm.results_dir,self.worm.cache_dir)

        
#        self.draw_text()
        
#DEBUG doesn't work but fixme later       
#        for channel in self.worm.channels.values():#This redraws the neurons in case they were loaded from .pkl
#            channel.set_small_scene_patch()
    
    def check_for_cube(self):
        self.has_atleast_one_local_view=False
        for scene_name in self.hash_volume.keys():
            scene=self.scene_dict[scene_name]
            if scene.is_local_view:
                self.has_atleast_one_local_view=True
        
        for scene_name in self.hash_volume.keys():
            scene=self.scene_dict[scene_name]
            if (not scene.is_local_view)  and (self.has_atleast_one_local_view):
                channel_name=self.hash_volume[scene_name]
                self.worm.channels[channel_name].display_grid_on_scene(scene)

        for key,seg in self.worm.segments.items():#if loaded from .pkl, draw neurons
            point=seg['marker']
            for channel in self.worm.channels.values():
                for scene in channel.drawn_scenes:
                    channel.draw_neuron_on_scene(point,scene)


    def sync_color_control(self):
#        for channel in self.worm.channels.values():
        channel=self.worm.segments.seg_channel
        self.sync_trait('volume_control',channel,'small_vol_module',mutual=True)
#            for module_alias in channel.vol_modules:
#                self.sync_trait('volume_control',channel,module_alias,mutual=True)

   
    def add_scene(self,name,scene):
        scene.name=name
        self.scene_dict[name]=scene
        self.add_trait(name,scene)
    
#    def draw(self):
#        self.current_mesh

#    def _name_changed(self):
#         self.current_neuron=self.worm.segments[self.name]
#         self.current_mesh=self.current_neuron['mesh']    


    current_name=Str('')
    def _current_name_changed(self):
        if self.current_name=='':
            return
        self.current_neuron=self.worm.segments[self.current_name]
        self.worm.segments.current_name=self.current_name
        if 'mesh' in self.current_neuron.keys():
            self.current_mesh=self.current_neuron['mesh']
        
        center=self.worm.segments.seg_channel.center.__array__()
        spacing=self.worm.segments.seg_channel.spacing
        scale=self.worm.segments.seg_channel.nm_voxel_shape
        p=np.ceil((self.current_neuron['marker']//scale)*scale)
        self.worm.center.set_value(p)#center view on marker we're talking about

    def _reset_current_mesh_fired(self):
        print 'using incomplete method _reset !!! '
#        self.clear_grids_from_figure()#Remove mesh from small scene
        self.worm.segments.seg_channel.clear_mesh()
        self.current_mesh.small_mesh.parent.parent.remove()
        del self.worm.segments[self.current_name]['mesh']#delete mesh
        self._seg_mode_fired()

#    def clear_mesh(self):
#        if not hasattr(self,'current_mesh'):
#            return
##        MScene=self.current_mesh.small_mesh.scene.mayavi_scene
##        MScene.remove_child(self.current_mesh.small_mesh)
#        scene=self.worm.segments.seg_channel.small_scene
#        MScene=scene.mayavi_scene
#        for child in MScene.children:
#            if child.name=='GridSource':
#                MScene.remove_child(child)
    
    next_marker=Button('next')
    def _next_marker_fired(self):
#        self.clear_mesh()
        self.current_name=self.worm.segments.next_key(self.current_name)
    previous_marker=Button('previous')
    def _previous_marker_fired(self):
#        self.clear_mesh()
        self.current_name=self.worm.segments.previous_key(self.current_name)
    last_marker=Button('last')
    def _last_marker_fired(self):
#        self.clear_mesh()
        self.current_name=self.worm.segments.last_key()

    delete_marker =Button()
    def _delete_marker_fired(self):
        del self.worm.segments._dict[self.current_name]
        self.current_name='' 
        for channel in self.worm.channels.values():
            channel.redraw_neurons(force_all=True)

    def _next_iteration_fired(self):
        print 'Display.py:_next_iteration_fired()'
#        offset=self.worm.segments.seg_channel.origin
        offset=np.array([0.0,0.0,0.0])
        self.current_mesh.update_zeroed_plot(offset)
        self.Average_Intensity=self.current_mesh.interior_intensity
        self.current_neuron['intensity']=self.Average_Intensity
#        if 'intensity' in self.current_neuron.keys():
#            self.current_neuron['intensity'][str(self.time)]=self.Average_Intensity
#        else:
#            self.current_neuron['intensity']=dict({str(self.time):self.Average_Intensity})


    seg_mode=Button('segment_this')
    def _seg_mode_fired(self):
        if self.current_name != '':
            self.worm.segments.draw_mesh_on_scene()#Defaults to segments.current_name
            self.current_mesh=self.current_neuron['mesh']
            self.reset_params()

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
    #debug: should be @on_trait_change here
    def _step_size_changed(self):
        self.current_mesh.tau=self.step_size    
    def reset_params(self):
        self.radius_best_guess=self.current_mesh.r_est
        self.emphasis_on_radius_guess=self.current_mesh.lamR
        self.emphasis_on_marker_guess=self.current_mesh.lamM
        self.smoothness=self.current_mesh.lamL
        self.step_size=self.current_mesh.tau
    #You should use this method if display depends on the number of instance
    #variables. However, these variables still need to be all declared before
    #config_traits(). Otherwise, you need to use a dynamic traits object

    def _debug_string_changed(self):
        try:
            print eval('self.'+self.debug_string+'print_traits()')
#            print self.a
        except:
            try:
                print eval('self.'+self.debug_string)
#                print self.a
            except:
                print 'no such member: ',self.debug_string
        self.a=eval('self.'+self.debug_string)
#        exec self.debug_string
        print self.a

    segmentation_parameters= VGroup(
                        HGroup(Item('next_iteration',show_label=False),spring,'Average_Intensity',),
                        'radius_best_guess',
                        'emphasis_on_radius_guess',
                        'emphasis_on_marker_guess',
                        'smoothness',
                        'step_size',
                        label='seg',
                    )

    side_panel=VGroup(
                        Item('save_segments',show_label=False),
                        
#                        visible_when='display_right_panel==True'
                    )

    
    update_trace=Button()
    def _update_trace_fired(self):
        print 'update trace fired'
        self.analyzer.update_data_for(self.current_name)
        print 'updated data'
        self.analyzer.update_plot_for(self.current_name)
        print 'updated plot'
    



    def default_traits_view(self):
#        scene_items=[VGroup(
#            Item(name,style='custom',show_label=False),
#            Item(name+'.downsample', show_label=False,visible_when=name+'.downsample!=-1'),
##            visible_when='name.downsample!=-1'
#                    )
#                for name in self.scene_dict.keys()]

        scene_items=[
            Item(name,style='custom',show_label=False) for name in self.scene_dict.keys()]
        
#        slider_navigation=Item('object.worm',style='custom',show_label=False,resizable=True),#Controls for adjusting center and radius
        slider_navigation=Item('object.worm',style='custom',resizable=True),#Controls for adjusting center and radius
        
        
        if self.show_trace:
            neuron_trace=VGroup(
                Item('update_trace',show_label=False),
                Item('object.analyzer.trace',style='custom',show_label=False, ),##NEURON TRACE
            )    
        else:
            neuron_trace=VGroup()
            
        traits_view = View(
            HGroup(
                VGroup(
                
                    HGroup(*scene_items),
#                    HSplit(
        #                Item('downsample', show_label=False,visible_when='downsample!=-1')
#                        Item('object.worm',style='custom',show_label=False),#Controls for adjusting center and radius
#                        slider_navigation,
                        
                        HGroup(
                            HGroup(
                                
                                HGroup('',slider_navigation,label='explorer',show_border=True),
                                VGroup(
                                    Item('save_segments',show_label=False),
    #                                HGroup('previous_marker','next_marker',enabled_when="current_name!=''"),
                                    HGroup(Item('previous_marker',show_label=False),
                                           Item('next_marker',show_label=False),
                                            Item('last_marker',show_label=False),
                                            Item('delete_marker',show_label=False),
                                            ),

                                    HGroup(
                                        Item('seg_mode',show_label=False),
                                        Item('reset_current_mesh',show_label=False),
                                    ),
                                    'current_name',
                                    label='anno',
                                    
                                ),
            #                spring,
    #                        Item('show_right_panel',show_label=False),#"more"#used to display right panel
                                Include('segmentation_parameters'),#make popout later
                                
                                layout='tabbed',
    #                            resizable=True,
                                springy=True,
                            
                            ),
                        
#                        Item('object.trace',style='custom'),##NEURON TRACE
#                        Include('neuron_trace'),   ####panel for neuron trace
                        neuron_trace,   ####panel for neuron trace
#                        Item('object.analyzer.trace',style='custom',show_label=False),

                            
                        ),
#                        springy=True


#                    ),
                    HGroup(
#                        'sync',
                        Item('debug_string',show_label=False,editor=TextEditor(auto_set=False,enter_set=True,evaluate=str)),
                    
                    )            
#                    VGroup(
##                        Item('object.worm',label='FOV'),
#                        'display_right_panel',
#                    ),
        
                ),
                
#                VGroup(
#                    Include('side_panel'),
##                    'next_marker',
##                    Item('next_marker'),
##                    visible_when='display_right_panel==True',
#                ),
                
#                Item('volume_control',style='custom',height=250, width=100, show_label=False,resizable=True),
#                Item('volume_control',style='custom',height=250, width=100, show_label=False),
            ),
            resizable=True,
            )
        return traits_view

#    traits_view = View( 
#        HGroup(
##            Include('coordinate_control'),
#            VGroup(
#                Item('object.large_scene',style='custom',show_label=False),
#                Item('object.large_scene.downsample'),
##                Include('object.large_scene',style='custom'),
#            ),
#            VGroup(
#                Item('object.small_scene',style='custom',show_label=False),
##                Item('object.small_scene',style='custom'),
##                Item('object.small_scene.downsample',visible_when='object.small_scene.downsample != -1'),
#                Include('dashboard'),
#
#
#                
#            ),
#            VGroup(
#                Item('object.worm',label='FOV'),
#                'display_right_panel',
#            ),
#
#            VGroup(
#    ##                    Item('cube.vol_module',style='custom',height=250, width=300, show_label=False,enabled_when='cube.vol_module.activated')
#    ##                    Item('cube.vol_module',style='custom',height=250, width=300, show_label=False)
#                        Item('volume_control',style='custom',height=250, width=100, show_label=False),
##                        spring,
#                           visible_when='display_right_panel==True',  
#                    ),
#        ),
#    resizable=True
#    )



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
#    vis=Display([11,11,11],I)

    vis=Display([11,11,11],None,None,I)
    
    
    from WormBox.BaseClasses import ApoReader
    apo=ApoReader([[22,22,22],[43,34,35]])
    apo.add_marker_list_to_dict(vis.worm.segments)
    
    
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
    
    