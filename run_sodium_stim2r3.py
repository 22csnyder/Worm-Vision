
import numpy as np
#from WormBox.BaseClasses import tEnvironment as Env
from WormBox.VisClasses import MultiChannelWorm
from WormScene import WormScene,Worm4DScene
from Display import Display

#StimOn: step232.8
#StimOff: step291
dataset='151125_1814_Vol_NaCl_10mM_30s_4min_W2 Region3'

cm_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151125_CM_NaCl_10mM_30s_4min_W2/TIFF/'
vol_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/TIFF/'#starts on 8#ends 6982 #465 timepoints
cache_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Cache/Region3/Cache'
save_results_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Cache/Region3/Results/'

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


def compute_median(array):
    s=array.shape[1]//2
    A=np.median(array[:,:s],axis=0)#Split up computation for memory considerations
    B=np.median(array[:,s:],axis=0)
    return np.vstack([A,B])
    
    

##Load data into channels
worm.add_channel('cm',cm_dir)#Hi Res 3D

worm.add_channel('mc_vol',vol_dir)
#worm.add_channel('vol',vol_dir)


import dask.array as da


from skimage import io
mcR3=io.imread('/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Cache/Region3/corrected_and_filledHead3.tif')
chunks=(mcR3.shape[0]*(1,),mcR3.shape[1]*(1,),(mcR3.shape[2],),(mcR3.shape[3],))
daMC=da.from_array(mcR3,chunks=chunks)


#from run_sodium_stim2
pos=np.array([1,241,98])
length=np.array([7,97,91])
mclen=mcR3.shape[1:]

worm['mc_vol'].daX=daMC

###array doesn't suport item assignment:
#worm['mc_vol'].daX[:,pos[0]:pos[0]+mclen[0],pos[1]:pos[1]+mclen[1],pos[2]:pos[2]+mclen[2]] = daMC[:]
#worm['mc_vol'].daX[:,pos[0]+mclen[0]:pos[0]+length[0],pos[1]+mclen[1]:pos[1]+length[1],pos[2]+mclen[2]:pos[2]+length[2]]=0


worm.add_channel_derived_from('mc_vol',with_process=compute_median, named='med_mc')



gui=Display()
gui.worm=worm


##Region 3 motion correction
#gui.worm.center.set_value([7133, 86925, 42966])
#gui.worm.radius.set_value([5498, 14418, 13546])



######The order added here determines the left->right order of scenes
gui.add_scene('LocalCM', WormScene(is_local_view=True) )
gui.add_scene('MedMotCor', WormScene(is_local_view=True) )
gui.add_scene('MotionCorrection', Worm4DScene(is_local_view=True) )



worm.segmentation_channel_is('med_mc')
worm.extract_time_series_data_for('mc_vol')



gui.hash_volume=dict({
#    'Whole_Worm':'cm',
#    'Time_Series':'vol',
    'LocalCM':'cm',
    'MedMotCor':'med_mc',
    'MotionCorrection':'mc_vol',
#    'Max_Box_Average':'mba30',
#    'Med_Local':'med',
#    'Time_Series':'vol',
#    'Before_Stim':'before',
#    'During_Stim':'during',
#    'After_Stim':'after',
    })



gui.start()