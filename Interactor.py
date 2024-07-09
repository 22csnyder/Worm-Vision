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
import numpy as np
from mayavi import mlab


import time
from scipy.ndimage import zoom

from traits.api import Array,NO_COMPARE,Enum

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
#from WormBox.BaseClasses import Environment#,Nucleus
from WormBox.WormSources2 import MultiVolume,CubeMesh,SphereMesh
import collections
import dill
from mayavi.modules.volume import Volume
from traits.has_dynamic_views import HasDynamicViews, DynamicView,DynamicViewSubElement

from apptools.scripting.api import Recorder,recordable,set_recorder
from tvtk.api import tvtk
from WormBox.BaseClasses import tWorm

#import ipdb

class constant_camera_view(object):
    def __init__(self):
        pass
    def __enter__(self):
        self.orig_no_render=mlab.gcf().scene.disable_render
        if not self.orig_no_render:
            mlab.gcf().scene.disable_render=True
        cc=mlab.gcf().scene.camera
        self.orig_pos=cc.position
        self.orig_fp=cc.focal_point
        self.orig_view_angle=cc.view_angle
        self.orig_view_up=cc.view_up
        self.orig_clipping_range=cc.clipping_range
    def __exit__(self,t,val,trace):
        cc=mlab.gcf().scene.camera
        cc.position=self.orig_pos
        cc.focal_point=self.orig_fp
        cc.view_angle=self.orig_view_angle
        cc.view_up=self.orig_view_up
        cc.clipping_range=self.orig_clipping_range
        
        if not self.orig_no_render:
            mlab.gcf().scene.disable_render=False
        if t!=None:
            print t,val,trace
#            ipdb.post_mortem(trace)
            



def ButtonEvent(obj,event):
    if event== "RightButtonPressEvent":
        print 'right button was pressed'
	        
# Some logic to pick on click but no move
class MvtPicker(object):
    mouse_mvt = False
#    def __init__(self,scene,vfpicker,npicker):
#        self.virtual_finger_picker = vfpicker
    def __init__(self,scene,picker):
        self.picker = picker
        self.scene=scene
    def on_right_button_press(self, obj, evt):
    
        x, y = obj.GetEventPosition()
        self.picker.pick((x, y, 0), self.scene.renderer)
        
#        print 'on_button_press'
#        print 'obj:'
#        print obj.__dir__()
#        print dir(obj)
#        print obj.GetLastEventPosition()
#        print obj.GetPicker()
#        print 'obj end'
#        print ''
#        z,y,x=obj.GetPickPosition()    
        
        
#        self.mouse_mvt = False
#        print x,y
#        self.mouse_mvt = False
#    def on_mouse_move(self, obj, evt):
#        self.mouse_mvt = True
#    def on_button_release(self, obj, evt):###Couldn't get button_release to work###
#        print 'on_button_release'
#        if not self.mouse_mvt:
#            x, y = obj.GetEventPosition()
#            self.picker.pick((x, y, 0), self.scene.renderer)
#        self.mouse_mvt = False
#        print x,y


class Interactor(HasTraits):
#    scene=Instance(klass=MlabSceneModel,kw=dict(point_smoothing=False))
    scene=Instance(MlabSceneModel,())
    
    debug_string=Str('')

    def cbk(self,vtk_obj,event):
#        hotkeys=dict(toggle_mesh='t')
        print vtk_obj.GetClassName(),';',event,';',vtk_obj.GetKeyCode();vtk_obj.GetShiftKey()
        self.cbk_vtk_obj=vtk_obj
        self.event=event
#        if vtk_obj.GetKeyCode()==hotkeys(['toggle_mesh']):       
#            self._toggle_mesh_visibility_fired()
            
    def __init__(self,*worm_args):
        if len(worm_args)==1 and isinstance(worm_args[0],tWorm):#if passed tWorm just go with it
            self.worm=worm_args[0]
        else:
            self.worm=tWorm(*worm_args)#otherwise initialize another one
        
        
        HasTraits.__init__(self)
        
        self.t0=time.time()


#        self.Patch=patch

#        self.scene.mayavi_scene.add_child(self.array_src)
#        self.array_src=ArraySource()        
#        self.array_src.scalar_data=self.Patch
#        self.vol_module=Volume()
#        self.array_src.add_module(self.vol_module)
#        self.volume=self.vol_module.actors[0]

        self.worm.display_on_scene(self.scene)

        

#        self.volume.pickable=0
#        self.volume.pickable_=0
        
        
    @on_trait_change('scene.activated')
    def scene_activated(self):
        print 'scene is activated'
        self.camera=self.scene.scene.camera
        iren=self.scene.scene.interactor
        self.pointpicker=self.scene.scene.picker.pointpicker
#        mvt_picker = MvtPicker(self.scene.mayavi_scene.scene,self.pointpicker)
#        iren.add_observer('RightButtonPressEvent',mvt_picker.on_button_press)
        iren.add_observer('RightButtonPressEvent',self.on_right_button_press)


#        iren.add_observer('LeftButtonPressEvent',self.on_left_button_press)###was going to make a double-click
        
        
#        iren.add_observer('RightButtonPressEvent',self.picker_callback)
        self.pointpicker.add_observer('EndPickEvent', self.picker_callback)  
        
        iren.add_observer("KeyPressEvent",self.cbk)

#    def neuron_picker_callback(picker):
#        """ Picker callback: this get called when on pick events.
#        """
#        
#        if picker.actor in red_glyphs.actor.actors:
#            # Find which data point corresponds to the point picked:
#            # we have to account for the fact that each data point is
#            # represented by a glyph with several points
#            point_id = picker.point_id/glyph_points.shape[0]
#            # If the no points have been selected, we have '-1'
#            if point_id != -1:
#                # Retrieve the coordinnates coorresponding to that data
#                # point
#                x, y, z = x1[point_id], y1[point_id], z1[point_id]
#                # Move the outline to the data point.
#                outline.bounds = (x-0.1, x+0.1,
#                                  y-0.1, y+0.1,
#                                  z-0.1, z+0.1)

    def on_right_button_press(self, obj, evt):
        x, y = obj.GetEventPosition()
        self.pointpicker.pick((x, y, 0), self.scene.mayavi_scene.scene.renderer)
    
#    def on_left_button_press(self,obj,evt):
#        print 'lbp'
#        if time.time()-self.t0<1.0:###also check if time was too short!!
#            x, y = obj.GetEventPosition()
#            self.volume.pickable=0
#            self.pointpicker.pick((x, y, 0), self.scene.mayavi_scene.scene.renderer)
#            self.volume.pickable=1
#        self.t0=time.time()


    def picker_callback(self,picker_obj, evt):
#    def virtual_finger_picker_callback(self,picker_obj, evt):
        print 'picker was called back'
        self.picker_obj = tvtk.to_tvtk(picker_obj)
        print 'picker_obj'
        
        if self.picker_obj.view_prop is None:
            print 'Warning: click was not made on Object'
            return


#        print self.picker_obj.mapper_position
#        print         

        q1=np.array(self.picker_obj.mapper_position)
        q0=np.array(self.camera.position)
        
        d=q1-q0
        
        a0=q1-d/2
        a1=q1+d/2
        
        vfname='virtual finger'
        with constant_camera_view():
            mlab.plot3d([a0[0],a1[0]],[a0[1],a1[1]],[a0[2],a1[2]],tube_radius=.5,name=vfname,color=(1,0,0))
        mlab.move(3,1,-1.2)

        vflines=[c for c in self.scene.mayavi_scene.children if c.name==vfname]

        if len(vflines)==1:
            self.p0=q1
            self.m0=d
#            self.line1=np.array([d,q1]).transpose()
        elif len(vflines)==2:##This is the second line. Ready to make point
            self.p1=q1
            self.m1=d
            Y=np.array([np.dot(self.m0,self.p1-self.p0),np.dot(self.m1,self.p1-self.p0)]).transpose()
            A=np.array([[np.dot(self.m0,self.m0),-np.dot(self.m0,self.m1)],
                         [np.dot(self.m0,self.m1),-np.dot(self.m1,self.m1)]])
            
            Ans=np.linalg.solve(A,Y)
            alp,bet=Ans
            
            self.point=np.round(0.5*(  self.p0+alp*self.m0  +  self.p1+bet*self.m1  ))
            point=self.point
            with constant_camera_view():
                
                self.nodes=mlab.points3d(point[0],point[1],point[2],name='new marker',color=(1,0,0))            
                self.nodes.glyph.glyph.scale_factor=5
                
                for child in vflines:
                    self.scene.mayavi_scene.remove_child(child)
            
                    
            
            
        
#        print self.camera
        
        
        
        
        
            
            
#        print self.picker_obj.selection_point#Global point
        
#        picked = picker_obj.actors
#    #    if mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
#        
#        # m.mlab_source.points is the points array underlying the vtk
#        # dataset. GetPointId return the index in this array.
#        x_, y_ = np.lib.index_tricks.unravel_index(picker_obj.point_id,
#        r.shape)
#        print "Data indices: %i, %i" % (x_, y_)
#        n_x, n_y = r.shape
#        cursor.mlab_source.set(x=np.atleast_1d(x_) - n_x/2.,
#                               y=np.atleast_1d(y_) - n_y/2.)
#        cursor3d.mlab_source.set(x=np.atleast_1d(x[x_, y_]),
#                                 y=np.atleast_1d(y[x_, y_]),
#                                z=np.atleast_1d(z[x_, y_]))
        
        
        
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
        
            
#        self.a=eval('self.'+self.debug_string)
##        exec self.debug_string
#        print self.a
        
            
    traits_view = View( 
        VGroup(
            Item('scene',editor=SceneEditor(scene_class=MayaviScene),height=250, width=300, show_label=False),
            Item('debug_string',show_label=False,editor=TextEditor(auto_set=False,enter_set=True,evaluate=str))
            )
    )


    #x_, y_, z_ = picker_obj.pick_position
    #cursor3d.mlab_source.set(x=np.atleast_1d(x_),
    # y=np.atleast_1d(y_),
    # z=np.atleast_1d(z_))

################################################################################

        

if __name__=='__main__':


    path='/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/patches/patch94.tif'
    
    #I=io.imread(patchpath)
    I0=io.imread(path)
    I=I0.astype(np.float)
    I*=(255/I.max())
    
    
    I=I[:,10:65,:]
    
    
#    w=tWorm(I)
    
    vis=Interactor(I)
    vis.configure_traits()






    
#    src=mlab.pipeline.scalar_field(I)
#    mlab.pipeline.iso_surface
    
    
#    mlab.pipeline.volume(mlab.pipeline.scalar_field(I))
    
                    
#    mlab.show()



# Recorded script from Mayavi2
#from numpy import array
#try:
#    engine = mayavi.engine
#except NameError:
#    from mayavi.api import Engine
#    engine = Engine()
#    engine.start()
#if len(engine.scenes) == 0:
#    engine.new_scene()
# ------------------------------------------- 
#scene = engine.scenes[0]
#scene.scene.camera.position = [-54.70559013996013, 165.46101960195361, 191.80265544675842]
#scene.scene.camera.focal_point = [18.0, 38.0, 38.0]
#scene.scene.camera.view_angle = 30.0
#scene.scene.camera.view_up = [-0.28428706412651628, -0.79990570398465866, 0.52851842910454272]
#scene.scene.camera.clipping_range = [101.4551551115522, 352.96229077046036]
#scene.scene.camera.compute_view_plane_normal()
#scene.scene.render()
#scene.scene.camera.position = [-69.973764069351759, 192.22783371836385, 224.10121309057772]
#scene.scene.camera.focal_point = [18.0, 38.0, 38.0]
#scene.scene.camera.view_angle = 30.0
#scene.scene.camera.view_up = [-0.28428706412651628, -0.79990570398465866, 0.52851842910454272]
#scene.scene.camera.clipping_range = [145.64927293963638, 398.27242167501134]
#scene.scene.camera.compute_view_plane_normal()
#scene.scene.render()
#scene.scene.disable_render = True
#engine.remove_scene(scene.scene)
## ------------------------------------------- 
#from mayavi.tools.show import show
#show()







    
    #engine=Engine()
    #if len(engine.scenes)==0:
    #    engine.new_scene()
    #
    #scene=engine.scenes[0]
    
    
#        
#    s=mlab.pipeline.scalar_field(I)
#    v=mlab.pipeline.volume(s)
#    
#    
#    mlab.show()

#scene=mlab.gcf()


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
    
    
