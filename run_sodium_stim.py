
import numpy as np
#from WormBox.BaseClasses import tEnvironment as Env
from WormBox.VisClasses import MultiChannelWorm
from WormScene import WormScene,Worm4DScene
from Display import Display


dataset='151112_CM_NaCl_10mM_30s_no_anes_W2'

cm_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151112_CM_NaCl_10mM_30s_no_anes_W2/TIFF'
vol_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151112_1839_Vol_NaCl_10mM_30s_no_anes_W2/TIFF'#starts on 61#ends 3495#229 timepoints

cache_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151112_NaCl_10mM_30s_no_anes_W2/Cache/'
save_results_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151112_NaCl_10mM_30s_no_anes_W2/Results/'


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
worm.extract_time_series_data_for('vol')

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
gui.add_scene('LocalCM', WormScene(is_local_view=True) )
gui.add_scene('Med_Local', WormScene(is_local_view=True) )
#gui.add_scene('Med_Global', WormScene(is_local_view=True) )
#gui.add_scene('Local_Vol', WormScene(is_local_view=True) )
gui.hash_volume=dict({
    'Whole_Worm':'cm',
#    'Time_Series':'vol',
    'LocalCM':'cm',
    'Med_Local':'med',
#    'Med_Global':'med',
#    'Local_Vol':'vol',
    })


#worm.segments_filename='Segments_z32.pkl'
#gui.add_scene('Whole_Worm_z32', WormScene(downsample=4) )
#gui.add_scene('Zoom_Med_time_series_z32', WormScene(is_local_view=True) )
#gui.hash_volume=dict({
#    'Whole_Worm_z32':'med32',
#    'Zoom_Med_time_series_z32':'med32',})


#worm.segments_filename='Segments_z40.pkl'
#gui.add_scene('Whole_Worm_z40', WormScene(downsample=4) )
#gui.add_scene('Zoom_Med_time_series_z40', WormScene(is_local_view=True) )
#gui.hash_volume=dict({
#    'Whole_Worm_z40':'med40',
#    'Zoom_Med_time_series_z40':'med40',})

#worm.segments_filename='Segments_z48.pkl'
#gui.add_scene('Whole_Worm_z48', WormScene(downsample=4) )
#gui.add_scene('Zoom_Med_time_series_z48', WormScene(is_local_view=True) )
#gui.hash_volume=dict({
#    'Whole_Worm_z48':'med48',
#    'Zoom_Med_time_series_z48':'med48',})
    


    
#####Compare all at once (qualitative)#####
#gui.add_scene('Whole_Worm_High_Res',WormScene(downsample=5))
#gui.add_scene('Zoom_High_Res',WormScene(is_local_view=True))
#gui.add_scene('Zoom_Med_time_series_z24', WormScene(is_local_view=True) )
#gui.add_scene('Zoom_Med_time_series_z32', WormScene(is_local_view=True) )
#gui.add_scene('Zoom_Med_time_series_z40', WormScene(is_local_view=True) )
#gui.add_scene('Zoom_Med_time_series_z48', WormScene(is_local_view=True) )
#gui.hash_volume=dict({
#    'Whole_Worm_High_Res':'cm',
#    'Zoom_High_Res':'cm',
#    'Zoom_Med_time_series_z24':'med24',
#    'Zoom_Med_time_series_z32':'med32',
#    'Zoom_Med_time_series_z40':'med40',
#    'Zoom_Med_time_series_z48':'med48',
#    })


gui.start()


dont_continue_past_here_on_accident

#display.configure_traits()

#import dill
#fs24='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151111_Vol_Agarpad_W1-1_y334/Results/Segments_z24.pkl'
#with open(fs24,'rb') as handle:
#    s24=dill.load(handle)
#fs32='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151111_Vol_Agarpad_W1-1_y334/Results/Segments_z32.pkl'
#with open(fs32,'rb') as handle:
#    s32=dill.load(handle)
#fs40='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151111_Vol_Agarpad_W1-1_y334/Results/Segments_z40.pkl'
#with open(fs40,'rb') as handle:
#    s40=dill.load(handle)
#fs48='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151111_Vol_Agarpad_W1-1_y334/Results/Segments_z48.pkl'
#with open(fs48,'rb') as handle:
#    s48=dill.load(handle)



Intensity=dict()
#MedianIntensity=dict()
#LowerIntensity=dict()
segments=gui.worm.segments
vol=gui.worm['vol']

time_points=range(  len(vol.X) )


#for neuron_name in segments.keys()[:3]:
for neuron_name in segments.keys():
    current_neuron=segments[neuron_name]
    if not 'mesh' in current_neuron.keys():
        continue
    current_mesh=current_neuron['mesh']
    
    I_seg=current_mesh.return_segmented_image()
    
    median_patch=current_mesh.unscaled_patch
    ix,iy,iz=np.where(I_seg==1)
    
    m=median_patch[I_seg]
    inds=m.argsort()
    
    interior_size=I_seg.sum()
    
    percent=50.0
    topinds=np.round(percent/100*interior_size)

    best_inds=inds##just take all of them until I understand what I did
    
    bix=ix[best_inds]
    biy=iy[best_inds]
    biz=iz[best_inds]
    
    [x_pos,y_pos,z_pos]=current_neuron['marker']//vol.nm_voxel_shape  - vol.window_radius
    [x_len,y_len,z_len]=vol.window_shape
    try:##Still bad at handling edge cases
        y=[]
        for t in time_points:
            unscaled_patch=vol.X[t,x_pos:x_pos+x_len+1,y_pos:y_pos+y_len+1,z_pos:z_pos+z_len+1].compute()
            score=np.mean(unscaled_patch[bix,biy,biz])
            y.append(score)
    except:
        continue
    Intensity[neuron_name]=np.array(y)

import dill
with open(worm.results_dir+'t151123_Intensity.pkl','wb') as handle:
    dill.dump(Intensity,handle)


#with open(worm.results_dir+'t151123_Intensity.pkl','wb') as handle:
#    dill.dump(Intensity,handle)


import matplotlib.pyplot as plt

from produce_correlation_plots import plot_correlation_matrix,dict2mat

_Data,_Labels=dict2mat(Intensity)
x_pos=np.array([segments[name]['marker'][2] for name in _Labels])
inds=x_pos.argsort()
Data=_Data[inds];Labels=_Labels[inds]
plot_correlation_matrix(Data,Labels)
plt.title('Volume Intensity over Time 151112 NaCl W2')
plt.show()



from WormBox.WormPlot import getsquarebysideaxes


L=6
t=range(len(Intensity.values()[0]))
fig,axes=getsquarebysideaxes(L)
for i in range(L):
    ax=axes[i]
    key=Intensity.keys()[i]
    y=Intensity.values()[i]
    ax.plot(t,y,'b')
    ax.set_title(key)
    ax.set_xlim(0,len(t))
plt.show()





