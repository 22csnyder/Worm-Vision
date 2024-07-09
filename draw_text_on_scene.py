# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:55:11 2015

@author: csnyder
"""


from WormBox.VisClasses import MultiChannelWorm
from WormScene import WormScene,Worm4DScene
from Display import Display
import numpy as np

dataset='151112_CM_NaCl_10mM_30s_no_anes_W2'

#cm_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151112_CM_NaCl_10mM_30s_no_anes_W2/TIFF'
#vol_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151112_1839_Vol_NaCl_10mM_30s_no_anes_W2/TIFF'#starts on 61#ends 3495#229 timepoints
#cache_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151112_NaCl_10mM_30s_no_anes_W2/Cache/'
#save_results_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151112_NaCl_10mM_30s_no_anes_W2/Results/'

cm_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151125_CM_NaCl_10mM_30s_4min_W2/TIFF/'
vol_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/TIFF/'#starts on 8#ends 6982 #465 timepoints
cache_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Cache/'
save_results_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Results/'



um_window_shape=np.array([10.0,10.0,10.0])
nm_window_shape=1000*um_window_shape

#A worm can have multiple data channels, but only 1 set of neuron segments
worm=MultiChannelWorm(nm_window_shape,cm_dir)
worm.cache_dir=cache_dir
worm.results_dir=save_results_dir

################################################
#worm.segments_filename='Segments_cm_z32.pkl'#check lower
#worm.segments_filename='NonSave'
################################################


##Load data into channels
worm.add_channel('cm',cm_dir)#Hi Res 3D
worm.add_channel('vol',vol_dir)
#worm.add_channel('bad_vol24',vol24_dir)
#worm.add_channel('bad_vol32',vol32_dir)#first frame at start of each vol is bad
#worm.add_channel('bad_vol40',vol40_dir)
#worm.add_channel('bad_vol48',vol48_dir)

#def compute_median(array):
#    s=array.shape[1]//2
#    A=np.median(array[:,:s],axis=0)#Split up computation for memory considerations
#    B=np.median(array[:,s:],axis=0)
#    return np.vstack([A,B])



def compute_median(array):
    s=array.shape[1]//2
    A=np.median(array[:,:s],axis=0)#Split up computation for memory considerations
    B=np.median(array[:,s:],axis=0)
    
    return np.vstack([A,B])

##Create channel derived from other data
#throw out first plane of timepoint and compute median(array):
def mirror_first_and_median(array):
    s=array.shape[1]//2
    A=np.median(array[:,:s],axis=0)#Split up computation for memory considerations
    B=np.median(array[:,s:],axis=0)
    C=np.vstack([A,B])
    C[0,:,:]=C[1,:,:]#discard first frame without messing up spacing
    return C


worm.add_channel_derived_from('vol',with_process=compute_median, named='med')

worm.segmentation_channel_is('med')


gui=Display()
gui.worm=worm

######clean generic good starting point
gui.worm.center.set_value([18725,44775,165123])#nm
#gui.worm.radius.set_value([10638,15416,25508])#nm


###maybe z important02 med48#### 
#gui.worm.center.set_value([57040,20980,440622])
#gui.worm.radius.set_value([10940,12548,9894])

##############Comment/Uncomment Different Datasets################

#worm.segments_filename='Segments_cm.pkl'
#gui.add_scene('Whole_Worm_High_Res',WormScene(downsample=5))
#gui.add_scene('Zoom_High_Res',WormScene(is_local_view=True))
#gui.hash_volume=dict({
#    'Whole_Worm_High_Res':'cm',
#    'Zoom_High_Res':'cm',})

#worm.segments_filename='Segments_z24.pkl'

##DEBUG## only one local scene for now until further notice because I broke it
gui.add_scene('Whole_Worm', WormScene(downsample=6) )
#gui.add_scene('Time_Series', Worm4DScene() )
#gui.add_scene('LocalCM', WormScene(is_local_view=True) )
#gui.add_scene('Med_Local', WormScene(is_local_view=True) )
#gui.add_scene('Med_Global', WormScene(is_local_view=True) )
#gui.add_scene('Local_Vol', WormScene(is_local_view=True) )
gui.hash_volume=dict({
    'Whole_Worm':'cm',
#    'Time_Series':'vol',
#    'LocalCM':'cm',
#    'Med_Local':'med',
#    'Med_Global':'med',
#    'Local_Vol':'vol',
    })
    
scale=gui.worm['cm'].nm_voxel_shape
from mayavi import mlab
mlab.figure()
Arr=gui.worm['cm'].X.compute()[0]

dar=Arr[:,::5,::5]
izi,iyi,ixi=np.indices(dar.shape)
s=mlab.pipeline.scalar_field(izi,5*iyi,5*ixi,dar)
v=mlab.pipeline.volume(s)

key,neu=gui.worm.segments.items()[0]

for key,neu in gui.worm.segments.items():
    p=neu['marker']//scale
    mlab.text3d(x=p[0],y=p[1],z=p[2],text=key,orient_to_camera=True,line_width=10,scale=[30,30,30])
    nodes=mlab.points3d(p[0],p[1],p[2],name='new marker',color=(0,1,0))
    nodes.glyph.glyph.scale_factor=20





