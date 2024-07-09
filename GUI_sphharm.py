# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:53:29 2015

@author: csnyder
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:43:56 2015

@author: csnyder
"""

import numpy as np
from mayavi import mlab
from traits.api import HasTraits, Range, Instance, \
                    on_trait_change
from traitsui.api import View, Item, HGroup,VGroup
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import \
                    MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from traits.api import Button

from traits.api import Float,Int




from obj_polar_nucleus3D import PolarContour
from WormBox.BaseClasses import Environment,Nucleus
import time


from obj_polar_nucleus3D import SphereParams

#spherule.do_iter()
#Angle=np.mgrid[-np.pi:np.pi:theta_spacing,0:np.pi:np.complex(0,phi_steps)]
#Izi,Ixi,Iyi=np.indices(I.shape)

class TimeInterval():
    def __init__(self,name):
        self.t0=time.time()
        self.name=name
    def measure(self):
        self.t1=time.time()
        self.length=self.t1-self.t0
        return self.length            
    

class TimeKeeper():
    schedule=[]
    
    def add_event(self,time_interval):
        self.schedule.append(time_interval)
    def print_itinerary(self):
        for sch in self.schedule:
            print sch.name,': ',sch.length
            
            
class Visualization(HasTraits,PolarContour):

    button=Button('next_iteration')
    scene      = Instance(MlabSceneModel,())


    
    Average_Intensity=Float()
    radius_best_guess=Range(low=1.0,high=15.0,value=SphereParams.r_est)        
    emphasis_on_radius_guess=Range(low=0.0,high=30.0,value=SphereParams.lamR)
    smoothness=Range(low=0.0,high=500.0,value=SphereParams.lamL)
     
    emphasis_on_marker_guess=Range(low=0.0,high=500.0,value=SphereParams.lamM)
#    go_button=Button('Start/Stop')

    
    def __init__(self,I,Iscale,spacing,c_est):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        
        PolarContour.__init__(self,I,Iscale,spacing,c_est)

        self.timekeep=TimeKeeper()

        AngleWrap=np.hstack([self.Angle,self.Angle[:,0,:][:,np.newaxis,:]])
        self.theta_mesh,self.phi_mesh=AngleWrap

        mz,mx,my=self.update_mesh_coords()
        self.mesh=self.scene.mlab.mesh(mz,mx,my,representation='wireframe',color=(1,1,1))
        
        self.Average_Intensity=self.interior_intensity        
        
        self.sIzi,self.sIxi,self.sIyi=spacing[0]*self.Izi,spacing[1]*self.Ixi,spacing[2]*self.Iyi     
        s=mlab.pipeline.scalar_field(self.sIzi,self.sIxi,self.sIyi,self.I)
        self.volume=mlab.pipeline.volume(s)
        self.volume.volume_property.scalar_opacity_unit_distance = 2.0

        self.center_point3d=mlab.points3d(spacing[0]*self.c_est[0],spacing[1]*self.c_est[1],spacing[2]*self.c_est[2],color=(0.9,0,0))


    def update_mesh_coords(self):
        RadiusWrap=np.vstack([self.Radius,self.Radius[0,:][np.newaxis,:]])#Add theta at 2pi in addition to at 0 
        mz= RadiusWrap*np.cos(self.phi_mesh) + self.Center[0]                         *self.spacing[0]
        mx= RadiusWrap*np.sin(self.phi_mesh)*np.cos(self.theta_mesh) + self.Center[1] *self.spacing[1]
        my= RadiusWrap*np.sin(self.phi_mesh)*np.sin(self.theta_mesh) + self.Center[2] *self.spacing[2]
        return mz,mx,my    

    @on_trait_change('radius_best_guess,emphasis_on_radius_guess,emphasis_on_marker_guess,smoothness')
    def update_params(self):
        self.r_est=self.radius_best_guess
        self.lamR=self.emphasis_on_radius_guess
        self.lamM=self.emphasis_on_marker_guess
        self.lamL=self.smoothness

    @on_trait_change('button')
    def update_plot(self):
        timeint=TimeInterval('update_plot')
        timeint2=TimeInterval('compute_iteration')
        for i in range(100):
#            self.do_iter()
            self.do_cython_iter()
        self.Average_Intensity=self.interior_intensity
#        print 'time to compute iteration',timeint2.measure()
        mz,mx,my=self.update_mesh_coords()
        self.mesh.mlab_source.set(x=mz,y=mx,z=my)
#        print 'time to update plot', timeint.measure()
        self.timekeep.add_event(timeint)
        self.timekeep.add_event(timeint2)

    

    main_scene=Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                    height=250, width=300, show_label=False)
    dashboard= VGroup(
                        'Average_Intensity',
                        'button',
                        'radius_best_guess',
                        'emphasis_on_radius_guess',
                        'emphasis_on_marker_guess',
                        'smoothness',
                    )

    # the layout of the dialog created
#    traits_view = View(main_scene,
#                    dashboard,
#                )
    
    aaa=3


