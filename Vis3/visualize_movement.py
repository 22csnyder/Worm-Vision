# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 05:59:12 2015

@author: csnyder
"""
import numpy as np


from movie_analysis import Worm,DirectoryResource
from volume_views3 import Vis

import ConfigParser
Config=ConfigParser.ConfigParser()

#from WormBox.WormSources2 import Environment


##Example of use
#ini_file=io_pal.read_dir+'../TIFF.ini'
#Config.read(ini_file)
#x_nm=Config.get('scaled image info','xvox_nm')#Example of use

#'C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/first 4 stacks/sc_150902_1703_Vol_Dev_II_No_Stimulus_W1_W1_ZIM294_L4_No_Stimulus_O20X_F630x150_P11008x500S188_00021.tiff'
#rd='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/first 4 stacks/'

#rd='C:/Users/cgs567/Documents/Corral/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/TIFF/'
#wd='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/'


dataset='150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/TIFF/'
#dataset='150821_1656_Vol_Img_Device_II_Glycerol_1M_W1/TIFF/'#Doesn't display well on global

if dataset=='150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/TIFF/':
    re_file='sc_150902_1703_Vol_Dev_II_No_Stimulus_W1_W1_ZIM294_L4_No_Stimulus_O20X_F630x150_P11008x500S188_*****.tiff'
    ###This marker goes with the line::worm.daX=worm.daX[:,:,194:357,40:276]
    ylim=[194,357];xlim=[40,276]
#    marker_list=[[79,57,8],[142,95,9],[45,71,3],[209,145,8]]
#    apo_file='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/150902_1703_Vol_W1_cropped at 194-357y_40-276x 150908chris.apo'
    apo_file='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/150902_1703_Vol_W1_cropped at 194-357y_40-276x 150908chris_reduced.apo'
    spacing=np.array([5.6,1.0,1.0]) 
elif dataset=='150821_1656_Vol_Img_Device_II_Glycerol_1M_W1/TIFF/':
    re_file='c_150821_1656_Vol_Device_II_Glycerol_1M_W1_Dev_II_Gly_1M_L4_W1_O20X_F630x150_P11008x500S194_*****.tiff'
#    ylim=[0,1000];xlim=[0,400]
    ylim=[210,450];xlim=[114,1000]
    
#    spacing=np.array([5.8,1.0,1.0])    
    spacing=np.array([6,1.0,1.0])    

io_pal=DirectoryResource(dataset)

worm=Worm()
worm.read_dir=io_pal.read_dir
worm.dask_read_data(re_file,15)
#
#
worm.daX=worm.daX[:,:,ylim[0]:ylim[1],xlim[0]:xlim[1]]#(t,z,y,x) format

#plane=9
#P=worm.plane_by_plane(plane)
#io.imsave(wd+'plane'+str(plane)+'.tif',P)


varWorm=Worm()
var=worm.daX.compute().astype(np.float)
var/=var.max()
varWorm.daX=np.sum(var**2,axis=0)[np.newaxis,:]#sum along time axis. pretend to have 1 timepoint




env=DirectoryResource(dataset)
env.write_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/Seg1/'
#neuron_window=np.array([75,75,35])
#neuron_window_shape=np.array([37,37,5])##It really does have to be Odd in all coords.New std.
neuron_window_shape=np.array([37,37,9])##It really does have to be Odd in all coords.New std.
#Debug
#neuron_window_shape=np.array([71,71,5])##It really does have to be Odd in all coords.New std.
#spacing=np.array([5.8,5.0,5.0])   

##Get shapes first from variance
var_vis=Vis(varWorm,env,spacing,neuron_window_shape)
var_vis.record_gui=False#Reimplement this for multiple N at once or when have apo file
var_vis.read_from_markerfile(apo_file)

var_vis.configure_traits()#develop meshes on var images
 
vis=Vis(worm,env,spacing,neuron_window_shape)
vis.record_gui=False#Reimplement this for multiple N at once or when have apo file
#vis.read_from_marker_list(marker_list)
vis.read_from_markerfile(apo_file)


vis.segments=var_vis.segments#copy over results



from mayavi import mlab
from mayavi.api import Engine
from mayavi.sources.api import ArraySource
from mayavi.modules.volume import Volume
def show(I):
    engine = Engine()
    engine.start()
    if len(engine.scenes) == 0:
        engine.new_scene()
    scene = engine.scenes[0]
    array_src=ArraySource(spacing=spacing)  
    array_src.scalar_data=I 
    scene.add_child(array_src)
    vol_module=Volume()
    volume=array_src.add_module(vol_module)
    return volume
    
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

vis.configure_traits()

#np.save('150902_1703_n14_var50_radius_n14.npy',vis.current_mesh.Radius)

#import matplotlib.pyplot as plt

def compute_centroid(ix,I):
    vec=zip(ix[0],ix[1],ix[2])
    center=np.zeros(3)
    wt=0
    for v in vec:
        center+= I[v] * np.array(v)
        wt+=I[v]
    return center.astype(np.float)/wt
        

Intensity=dict()

neuron_name='4'

time_points=range(  len(vis.worm.daX) )

for neuron_name in vis.segments.keys():    
    current_neuron=vis.segments[neuron_name]
    current_mesh=current_neuron['mesh']
    
    if 'quality' in vis.current_neuron.keys():
        if vis.current_neuron['quality'] == 'marker is bad':
            continue
    
    var_seg=current_mesh.return_segmented_image()
    var_ix=np.where(var_seg==1)
    
    y=[]
    for t in time_points:
        vis.time=t
        score=np.mean(vis.cube.unscaled_patch[var_seg])
        y.append(score)
    Intensity[neuron_name]=np.array(y)


import matplotlib.pyplot as plt



fig,axes=plt.subplots(6,5)
for i,name in enumerate(Intensity.keys()):
    ax=axes.ravel()[i]
    y=Intensity[name]
    ave=np.mean(y)
    output=y*100.0/y.max()
    ax.plot(time_points,output)
    ax.set_ylim([0,130])
    ax.set_xlim([0,len(time_points)])
    ax.set_title(name)
fig.subplots_adjust(hspace=.9)
plt.show()


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
