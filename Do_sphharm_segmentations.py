# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:33:20 2015

@author: csnyder
"""
from mayavi import mlab
from GUI_sphharm import Visualization
from WormBox.BaseClasses import Environment,Nucleus
from WormBox.ioworm import ls ,easy_contents,convert_index
import numpy as np
import skimage.io as io

from traits.api import HasTraits, Int,Button
from traitsui.api import View,HGroup,VGroup,Include,Item

#marker=12#Two small markers
#marker=14
#    marker=116#Easy

#    marker=57

from obj_polar_nucleus3D import SphereParams

##spacing=np.array([3.0,1.0,1.0])#300x100x100um #150226 Hi-res_40x
##spacing=np.array([2.0,1.0,1.0])#300x100x100um #150226 Hi-res_40x
spacing=np.array([1.0,1.0,1.0])


readdir='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/patches'
writedir='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/seg_patches'

env=Environment(readdir,writedir)


class SlideShow(Visualization):
    
    toggle_mesh_visibility=Button('on/off')    
    
    next_marker=Button()
    previous_marker=Button()
    m=Int(-1)
    
    save_segmentation=Button()
    
    def __init__(self,env):
        self.env=env
        self.marker_list,self.file_list=easy_contents(self.env.read_dir)
        self.n_markers=len(self.marker_list)

        self.jump_into_first_iter=False#Choose to displace iteration 0 or 1 when you start
        self.fast_advance=True#Go to next marker when saving
    
    def _toggle_mesh_visibility_fired(self):
        if hasattr(self,'mesh'):
            self.mesh.actor.actor.visibility= not self.mesh.actor.actor.visibility
    def start_with(self,marker,jump_into_first_iter=False):
        self.m=marker
        self._m_changed()
        self.jump_into_first_iter=jump_into_first_iter
        if jump_into_first_iter is True:
            self.update_plot()
        
    def reset_params(self):
        self.Center=self.c_est.copy()
        self.Aml=np.zeros(self.Aml.shape)
        
        self.radius_best_guess=SphereParams.r_est       
        self.emphasis_on_radius_guess=SphereParams.lamR
        self.emphasis_on_marker_guess=SphereParams.lamM
        self.smoothness=SphereParams.lamL
        self.Radius=np.zeros(self.Angle[0].shape)+self.r0
        self.Average_Intensity=0.0
    
    def _previous_marker_fired(self):
        if self.m is 0:
            print 'already at first marker'
            return
        self.m-=1
        self.update_image
    def _next_marker_fired(self):
        if self.m is self.marker_list[-1]:
            print 'already at last marker'
            return
        self.m+=1
        self.update_image()
    
    def _m_changed(self):
        self.update_image()
        
    def update_image(self):
        self.f=self.file_list[self.m]
        self.I=io.imread(self.f).astype(np.float)
        self.Iscale=self.I.max()
        self.I/=self.Iscale
        self.c_est=0.5*np.array(self.I.shape).astype(np.float)#z,x,y
        
        if not hasattr(self,'volume'):
#            print 'parent class is not yet init!'
            Visualization.__init__(self,self.I,self.Iscale,spacing,self.c_est)
        else:
#            print 'parent class already init'
            
            self.reset_params()
            self.volume.mlab_source.scalars=self.I
        
            mz,mx,my=self.update_mesh_coords()
            self.mesh.mlab_source.set(x=mz,y=mx,z=my)
        if self.jump_into_first_iter:
            self.update_plot()
            
#        self.vis=Visualization(I,Iscale,spacing,c_est)
    
    def _save_segmentation_fired(self):
                
        
        nucleus=Nucleus(self.env)
        nucleus.Patch=self.return_segmented_image()
        
        n_digits=len(str(self.n_markers-1))       
        string_index=convert_index(self.m,n_digits)

        nucleus.save('seg_Patch'+string_index)
        
        
        if self.fast_advance:
            self._next_marker_fired()
        else:
            print '...done saving'
        
    traits_view=View(VGroup(Include('main_scene'),
                    HGroup('m','previous_marker','next_marker'),
                    Include('dashboard'),'save_segmentation',
                    HGroup('toggle_mesh_visibility'))
                    )



if __name__=='__main__':
    slide=SlideShow(env)
    #slide.fast_advance=True
    #slide.start_with(40,jump_into_first_iter=True)
    slide.start_with(80,jump_into_first_iter=False)
    
    slide.configure_traits()






