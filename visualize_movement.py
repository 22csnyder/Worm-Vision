# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 05:59:12 2015

@author: csnyder
"""
import numpy as np


#from movie_analysis import Worm,DirectoryResource

from WormBox.BaseClasses import tWorm as Worm
from WormBox.BaseClasses import tEnvironment as Env


from volume_views import Vis

import ConfigParser
Config=ConfigParser.ConfigParser()



#smear=np.array([20,1.0,1.0])
#smear=np.array([6.1,1.0,1.0])
smear=np.array([2.1,1.0,1.0])


#dataset='151001_2012_Vol_Img_Glycerol_3ch_Dev_II_W3/TIFF/'
dataset='150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/TIFF/'
#dataset='150821_1656_Vol_Img_Device_II_Glycerol_1M_W1/TIFF/'#Doesn't display well on global

if dataset=='150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/TIFF/':
    ######take1 ######
#    re_file='sc_150902_1703_Vol_Dev_II_No_Stimulus_W1_W1_ZIM294_L4_No_Stimulus_O20X_F630x150_P11008x500S188_*****.tiff'
#    ###This marker goes with the line::worm.daX=worm.daX[:,:,194:357,40:276]
#    ylim=[194,357];xlim=[40,276]
##    marker_list=[[79,57,8],[142,95,9],[45,71,3],[209,145,8]]
##    apo_file='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/150902_1703_Vol_W1_cropped at 194-357y_40-276x 150908chris.apo'
#    apo_file='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/150902_1703_Vol_W1_cropped at 194-357y_40-276x 150908chris_reduced.apo'
#    spacing=np.array([5.6,1.0,1.0])
    ######take2 151013 ######    
    re_file='sc_150902_1703_Vol_Dev_II_No_Stimulus_W1_W1_ZIM294_L4_No_Stimulus_O20X_F630x150_P11008x500S188_*****.tiff'
    ylim=[194,357];xlim=[40,276]
    apo_file='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/t151012 flipped.apo'
#    spacing=np.array([5.6,1.0,1.0])
    read_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/TIFF/'
    write_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/Seg1/'
    
elif dataset=='150821_1656_Vol_Img_Device_II_Glycerol_1M_W1/TIFF/':
    re_file='c_150821_1656_Vol_Device_II_Glycerol_1M_W1_Dev_II_Gly_1M_L4_W1_O20X_F630x150_P11008x500S194_*****.tiff'
#    ylim=[0,1000];xlim=[0,400]
    ylim=[210,450];xlim=[114,1000]
#    spacing=np.array([5.8,1.0,1.0])    
    spacing=np.array([6,1.0,1.0])
    
elif dataset=='151001_2012_Vol_Img_Glycerol_3ch_Dev_II_W3/TIFF/':
    apo_file='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151001_CM_Glycerol_3ch_Dev_II_W3/151001_HQ1_Gly_3ch_D_II_W3_L4_O20X_scaled_to_Vol.apo'
    spacing=np.array([5.6,1.0,1.0]) 
    re_file='sc_151001_2012_Vol_Glycerol_3ch_Dev_II_W3_VOL_Gly_3ch_D_II_W3_L4_O20X_F630x150_P11008x500S192_*****.tiff'

    


###old style###
#io_pal=DirectoryResource(dataset)
#worm=Worm()
#worm.read_dir=io_pal.read_dir
#worm.dask_read_data(re_file,15)
#try:
#    worm.daX=worm.daX[:,:,ylim[0]:ylim[1],xlim[0]:xlim[1]]#(t,z,y,x) format
#except:
#    pass
#env=DirectoryResource(dataset)


#spacing=np.array([1.0,1.0,1.0])
#smear=([2.0,1.0,1.0])
#patch=np.zeros((21,51,51)).astype(np.double)
#c=np.array([10,25,25])
#z,y,x=np.indices(patch.shape)
#D=(z-c[0])**2+(y-c[1])**2+(x-c[2])**2
#patch[D<10**2]=2.0
#vis=Vis(patch,smear,spacing)
#vis.configure_traits()
#stop

worm=Worm(Env(read_dir,write_dir))
worm.get_ini_file()#sets the spacing
worm.dask_read_data()


MedX=np.load(worm.write_dir+'Median3D.npy')
#V=np.median(worm.daX,axis=0)
#np.save(worm.write_dir+'Median3D.npy',V)


medworm=Worm(Env(read_dir,write_dir))
medworm.get_ini_file()#sets the spacing
medworm.dask_read_data()
medworm.stack=MedX

#medworm.spacing=np.array([1,1,1])
#medworm.stack=np.ones(MedX.shape)

#neuron_window=np.array([75,75,35])
#neuron_window_shape=np.array([37,37,5])##It really does have to be Odd in all coords.New std.
neuron_window_shape=np.array([37,37,9])##It really does have to be Odd in all coords.New std.
#Debug
#neuron_window_shape=np.array([71,71,5])##It really does have to be Odd in all coords.New std.
#spacing=np.array([5.8,5.0,5.0])   

'''
vis=Vis(medworm,smear=smear,neuron_window_shape=neuron_window_shape)
#vis=Vis(worm,smear,neuron_window_shape)
vis.record_gui=False#Reimplement this for multiple N at once or when have apo file
#vis.read_from_marker_list(marker_list)
vis.read_from_markerfile(apo_file)
vis.configure_traits()
'''

#if vis.save_at_end:
#    vis.save_segments('t151014Segments')




#import dill
#segments=vis.segments
#with open(vis.env.write_dir+'t151014segments_dict2.pkl','wb') as handle:
#    dill.dump(segments,handle)
#with open(vis.env.write_dir+'t151014segments_dict.pkl','r') as handle:
#    a=dill.load(handle)

seg_pkl='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/Seg1/t151014segments_dict.pkl'
import dill
with open(seg_pkl,'r') as handle:
    segments=dill.load(handle)

window_radius=np.array(neuron_window_shape)//2
x_len,y_len,z_len=neuron_window_shape[::-1]
#    print 'z_pos:',z_pos
#    print 'z_len:',z_len



#Intensity=dict()
#MedianIntensity=dict()
LowerIntensity=dict()
time_points=range(  len(worm.stack) )
for neuron_name in segments.keys():    
    current_neuron=segments[neuron_name]
    current_mesh=current_neuron['mesh']
    
    if 'quality' in current_neuron.keys():
        if current_neuron['quality'] == 'marker is bad':
            continue
    
    var_seg=current_mesh.return_segmented_image()
    
    median_patch=current_mesh.unscaled_patch
    ix,iy,iz=np.where(var_seg==1)
    
    m=median_patch[var_seg]
    inds=m.argsort()
    
    interior_size=var_seg.sum()
    
    percent=25.0
    topinds=np.round(percent/100*interior_size)
    best_inds=inds[:(interior_size-topinds)]##actually the worst ones ### best_inds=inds[(interior_size-topinds) :]
    
    bix=ix[best_inds]
    biy=iy[best_inds]
    biz=iz[best_inds]
    
    center=current_neuron['marker'][::-1]
    x_pos=center[0]-window_radius[2]
    y_pos=center[1]-window_radius[1]
    z_pos=center[2]-window_radius[0]
    
    try:##Still bad at handling edge cases
        y=[]
        for t in time_points:
            unscaled_patch=worm.stack[t,x_pos:x_pos+x_len+1,y_pos:y_pos+y_len+1,z_pos:z_pos+z_len+1].compute()
            score=np.mean(unscaled_patch[bix,biy,biz])
            y.append(score)
    except:
        continue
    LowerIntensity[neuron_name]=np.array(y)
#    MedianIntensity[neuron_name]=np.array(y)
#    Intensity[neuron_name]=np.array(y)

#            score=np.mean(unscaled_patch[var_seg])

#with open(vis.env.write_dir+'t151014MeanIntensity.pkl','wb') as handle:
#    dill.dump(Intensity,handle)


#with open(vis.env.write_dir+'t151014MedianIntensity.pkl','wb') as handle:
#    dill.dump(MedianIntensity,handle)
with open(worm.write_dir+'t151014Lower75Intensity.pkl','wb') as handle:
    dill.dump(LowerIntensity,handle)




    


#        var_ix=np.where(var_seg==1)


#vis.configure_traits()
#if vis.save_status:
#    import dill
#    segments=vis.segments
#    with open(vis.env.write_dir+'segments.pkl','wb') as handle:
#        dill.dump(segments,handle)




#from mayavi import mlab
#from mayavi.api import Engine
#from mayavi.sources.api import ArraySource
#from mayavi.modules.volume import Volume
#def show(I):
#    engine = Engine()
#    engine.start()
#    if len(engine.scenes) == 0:
#        engine.new_scene()
#    scene = engine.scenes[0]
#    array_src=ArraySource(spacing=spacing)  
#    array_src.scalar_data=I 
#    scene.add_child(array_src)
#    vol_module=Volume()
#    volume=array_src.add_module(vol_module)
#    return volume
    
#    s=mlab.pipeline.scalar_field(vol)
#    v=mlab.pipeline.volume(s)
#    return v


#mlab.pipeline.volume(mlab.pipeline.scalar_field(vol))
'''
########Case study: 150902_1703 , '14'###########
vis.neuron_name='14'
vis._toggle_mesh_visibility_fired()

try:
    var=np.load('150902_1703_n14_var50.npy')
except:
    from volume_views3 import get_Patch
    marker=vis.current_neuron['marker'][::-1]
    window_shape=vis.neuron_window_shape[::-1]
    tpatch=[]
    for t in range(50):
        vis.time=t
        tpatch.append( get_Patch(marker,vis.cube.I,window_shape) )
    X=np.vstack([p[np.newaxis,:] for p in tpatch])
    time_ave=np.sum(X,axis=0)
    Xf=X.astype(np.float)/X.max()#12985 is Xmax
    var=np.sum(Xf*Xf,axis=0)

from mayavi.sources.api import ArraySource
from mayavi.modules.iso_surface import IsoSurface
array_src=ArraySource(spacing=spacing)  
array_src.scalar_data=var
array_src.name='var'
vis.small_scene.mayavi_scene.add_child(array_src)
iso_surface = IsoSurface()
vis.engine2.add_filter(iso_surface,array_src)
iso_surface.actor.property.representation = 'wireframe'

t=np.percentile(var.ravel(),95)#~1.8
iso_surface.contour.contours[0:1] = [t]
iso_surface.actor.property.opacity = 0.4



mlab.pipeline.image_plane_widget(vis.cube.array_src,plane_orientation='x_axes')

#Converge first to this envelope
boo_var=I=(var>t).astype(np.float)
vis.current_neuron['mesh'].I=np.ascontiguousarray(boo_var*200)
#vis.current_neuron['mesh'].Iscale=255.0
#vis.cube.array_src.scalar_data=boo_var*255.0
'''


#np.save('150902_1703_n14_var50_radius_n14.npy',vis.current_mesh.Radius)

#import matplotlib.pyplot as plt



#varWorm=Worm()
#var=worm.daX.compute().astype(np.float)
#var/=var.max()
#varWorm.daX=np.sum(var**2,axis=0)[np.newaxis,:]#sum along time axis. pretend to have 1 timepoint
##Get shapes first from variance
#var_vis=Vis(varWorm,env,spacing,neuron_window_shape)
#var_vis.record_gui=False#Reimplement this for multiple N at once or when have apo file
#var_vis.read_from_markerfile(apo_file)
#var_vis.configure_traits()#develop meshes on var images
#vis.segments=var_vis.segments#copy over results


#def compute_centroid(ix,I):
#    vec=zip(ix[0],ix[1],ix[2])
#    center=np.zeros(3)
#    wt=0
#    for v in vec:
#        center+= I[v] * np.array(v)
#        wt+=I[v]
#    return center.astype(np.float)/wt  
#
#Intensity=dict()
#neuron_name='4'
#time_points=range(  len(vis.worm.daX) )
#for neuron_name in vis.segments.keys():    
#    current_neuron=vis.segments[neuron_name]
#    current_mesh=current_neuron['mesh']
#    
#    if 'quality' in vis.current_neuron.keys():
#        if vis.current_neuron['quality'] == 'marker is bad':
#            continue
#    
#    var_seg=current_mesh.return_segmented_image()
#    var_ix=np.where(var_seg==1)
#    
#    y=[]
#    for t in time_points:
#        vis.time=t
#        score=np.mean(vis.cube.unscaled_patch[var_seg])
#        y.append(score)
#    Intensity[neuron_name]=np.array(y)
#
#
#import matplotlib.pyplot as plt
#
#
#fig,axes=plt.subplots(6,5)
#for i,name in enumerate(Intensity.keys()):
#    ax=axes.ravel()[i]
#    y=Intensity[name]
#    ave=np.mean(y)
#    output=y*100.0/y.max()
#    ax.plot(time_points,output)
#    ax.set_ylim([0,130])
#    ax.set_xlim([0,len(time_points)])
#    ax.set_title(name)
#fig.subplots_adjust(hspace=.9)
#plt.show()


#name='4'
#y=Intensity[name]
#ave=np.mean(y)
#output=y*100.0/ave
##    plt.figure()





#with open(vis.env.write_dir+'t150916 IntensityTrace.pkl','wb') as handle:
#    dill.dump(Intensity,handle)

'''
var_seg=vis.current_mesh.return_segmented_image()
var_ix=np.where(var_seg==1)
true_center=compute_centroid(var_ix,np.ones(var_seg.shape))

time_points=range(50)
intensity=[]
center_list=[]
deviation=[]
for t in time_points:
    vis.time=t
    center=compute_centroid(var_ix,vis.current_mesh.I)
    score=np.mean(vis.current_mesh.I[var_seg])*vis.current_mesh.Iscale
    intensity.append(score)
    deviation.append(np.linalg.norm(center-true_center))
    


import matplotlib.pyplot as plt
plt.plot(time_points,intensity)



fig,ax1=plt.subplots()
ax2=ax1.twinx()
ax1.plot(time_points,intensity,'b')
ax2.plot(time_points,deviation,'r')
ax1.set_ylabel('intensity',color='b')
ax2.set_ylabel('centroid deviation',color='r')

plt.show()

'''





#from mayavi.modules.iso_surface import IsoSurface
#engine = Engine()
#engine.start()
#if len(engine.scenes) == 0:
#    engine.new_scene()
#scene = engine.scenes[0]
#array_src=ArraySource(spacing=spacing)  
#array_src.scalar_data=var
#array_src.name='var'
#scene.add_child(array_src)
#vol_module=Volume()
#volume=array_src.add_module(vol_module)
#iso_surface = IsoSurface()
#engine.add_filter(iso_surface, array_src)
#iso_surface.actor.property.representation = 'wireframe'
#iso_surface.contour.contours[0:1] = [1.17415771582]




#np.save('150902_1703_n14_var50.npy',var)

#for neuron in self.segments.values():
#    marker=neuron['marker'][::-1]
#    window_shape=self.neuron_window_shape[::-1]
#    if 'mesh' in neuron.keys():
#        print 'mesh patch updated'
#        unscaled_patch=get_Patch(marker,self.cube.I,window_shape)
#        Iscale=unscaled_patch.max()
#        p0d=unscaled_patch.astype(np.float)
#        I=p0d/Iscale
#        neuron['mesh'].I=I
#        neuron['mesh'].Iscale=Iscale



#vis.configure_traits()
#if vis.save_status:
#    import dill
#    segments=vis.segments
#    with open(vis.env.write_dir+'segments.pkl','wb') as handle:
#        dill.dump(segments,handle)





#import dill
#segments=vis.segments
#with open(vis.env.write_dir+'t150916 segments.pkl','wb') as handle:
#    dill.dump(segments,handle)















#Plot neuron intensity over time

#from mayavi import mlab
#
#Ht=H[0].compute()
#
#s=mlab.pipeline.scalar_field(Ht)
#v=mlab.pipeline.volume(s)
#


#
#spacing=np.array([2.0,1.0,1.0])
##    spacing=np.array([1.0,1.0,1.0])
##I=io.imread(patchpath)
#I0=io.imread(path)
#print 'finished reading data'
#env=Environment(read_dir,write_dir)
#
#vis=Vis(wholeworm=I0,env=env,spacing=spacing)
#vis.read_from_markerfile(marker_file)
#
#vis.configure_traits()
#
#try:
#    vis.save_segments(global_offset)
#except:
#    vis.save_segments()
