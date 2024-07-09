
import numpy as np
#from WormBox.BaseClasses import tEnvironment as Env
from WormBox.VisClasses import MultiChannelWorm
from WormScene import WormScene,Worm4DScene
from Display import Display
from WormBox.ioworm import smakedirs


'''
Ki:
Recommendations of experiments for data analysis

+10 mM
1. 151214_1849_Vol_NaCl_10mM_30s_4min_W1
2. 151210_1640_Vol_NaCl_10mM_30s_4min_W2
3. 151215_2047_Vol_NaCl_10mM_30s_4min_W2
4. 151210_1601_Vol_NaCl_10mM_30s_4min_W1_2nd_try
5. 151215_2123_Vol_NaCl_10mM_30s_4min_W3

+50 mM
1. 151217_2109_Vol_NaCl_50mM_30s_4min_W5
2. 151217_1915_Vol_NaCl_50mM_30s_4min_W1
3. 151217_2034_Vol_NaCl_50mM_30s_4min_W4
4. 151217_2010_Vol_NaCl_50mM_30s_4min_W3
'''


## Head straight:
#data=dict(vol='/151210_1729_Vol_NaCl_10mM_30s_4min_W3', cm='/151210_CM_NaCl_10mM_30s_4min_W3')#14:6988
#data=dict(vol='/151214_1849_Vol_NaCl_10mM_30s_4min_W1', cm='/151214_CM_NaCl_10mM_30s_4min_W1') #6:6995
#data=dict(vol='/151214_1930_Vol_NaCl_10mM_30s_4min_W2', cm='/151214_1930_Vol_NaCl_10mM_30s_4min_W2') #3:6992
#data=dict(vol='/151217_1915_Vol_NaCl_50mM_30s_4min_W1', cm='/151217_CM_NaCl_50mM_30s_4min_W1') #2:6991  ###maybe no movement
#data=dict(vol='/151217_2109_Vol_NaCl_50mM_30s_4min_W5', cm='/151217_CM_NaCl_50mM_30s_4min_W5') #3:6992

##Head Curved:
data=dict(vol='/151217_2010_Vol_NaCl_50mM_30s_4min_W3',cm='/151217_CM_NaCl_50mM_30s_4min_W3') #7:6996


#StimOn: step232.8
#StimOff: step291

###DEFINE DATA###
#data=dict(vol='/151217_2109_Vol_NaCl_50mM_30s_4min_W5', cm='/151217_CM_NaCl_50mM_30s_4min_W5') #3:6992


#####   10mM    ######
#Motion correction didnt' work well here:
#data=dict(vol='/151125_1814_Vol_NaCl_10mM_30s_4min_W2', cm='/151125_CM_NaCl_10mM_30s_4min_W2') #3:6992

#The whole worm was annotated here by breaking it up into smaller pieces
#The head and neck are nice and straight here
#data=dict(vol='/151214_1849_Vol_NaCl_10mM_30s_4min_W1', cm='/151214_CM_NaCl_10mM_30s_4min_W1') #6:6995
#The head and neck are nice and straight here
#?#data=dict(vol='/151214_1930_Vol_NaCl_10mM_30s_4min_W2', cm='/151214_1930_Vol_NaCl_10mM_30s_4min_W2') #3:6992


#The head and neck are bent in the next 4
#data=dict(vol='/151210_1640_Vol_NaCl_10mM_30s_4min_W2', cm='/151210_CM_NaCl_10mM_30s_4min_W2') # 15:6989
#data=dict(vol='/151215_2047_Vol_NaCl_10mM_30s_4min_W2', cm='/151215_2047_Vol_NaCl_10mM_30s_4min_W2') # 15:6989
#data=dict(vol='/151210_1601_Vol_NaCl_10mM_30s_4min_W1_2nd_try', cm='/151210_CM_NaCl_10mM_30s_4min_W1') # 5:6994
#data=dict(vol='/151215_2123_Vol_NaCl_10mM_30s_4min_W3', cm='/151215_CM_NaCl_10mM_30s_4min_W3') # 12:6986


corral='/work/03176/csnyder/Corral'
cm_dir=corral+'/Ki-Confocal-2015'+data['cm']+ '/TIFF/'
vol_dir=corral+'/Ki-Confocal-2015'+data['vol']+'/TIFF/'

cache_dir=corral+'/Snyder/WormPics/Ki-Confocal-2015'+data['vol']+'/Cache/'
results_dir=corral+'/Snyder/WormPics/Ki-Confocal-2015'+data['vol']+'/Results/'
smakedirs(cache_dir,results_dir)

um_window_shape=np.array([10.0,10.0,10.0])
nm_window_shape=1000*um_window_shape

#A worm can have multiple data channels, but only 1 set of neuron segments
worm=MultiChannelWorm(nm_window_shape,cm_dir)
worm.cache_dir=cache_dir
worm.results_dir=results_dir

################################################
#worm.segments_filename='Segments_cm_z32.pkl'#check lower
#worm.segments_filename='NonSave'
################################################


##Load data into channels
worm.add_channel('cm',cm_dir)#Hi Res 3D
worm.add_channel('vol',vol_dir)


def compute_median(array):
    s=array.shape[1]//2
    A=np.median(array[:,:s],axis=0)#Split up computation for memory considerations
    B=np.median(array[:,s:],axis=0)
    return np.vstack([A,B])

from functools import partial
def during_interval(t0,t1,array):
    return compute_median(array[t0:t1])

def max_box_average(duration,array):
#    import dask.array as da
    rad=int(duration//2)
    L=len(array)    
    current_max=np.ones(array.shape[1:])*-np.inf
    for t in range(array.shape[0]):###Takes a really long time, nonparallel
        a=max(0,t-rad)
        b=min(t+rad,L)
        current_ave = np.mean(np.array(array[a:b]),axis=0)
        #"maximum" is elementwise max
        current_max=np.maximum(current_max,current_ave)
    return current_max
    
    
t0,t1=worm['vol'].stimulus_timepoints
#before_stim=partial(during_interval,0,t0)
#during_stim=partial(during_interval,t0,t1)
#after_stim=partial(during_interval,t1,None)

box30= partial(max_box_average,30)


#worm.add_channel_derived_from('vol',with_process=before_stim, named='before')
#worm.add_channel_derived_from('vol',with_process=during_stim, named='during')
#worm.add_channel_derived_from('vol',with_process=after_stim, named='after')

worm.add_channel_derived_from('vol',with_process=box30, named='mba30')

#worm.add_channel_derived_from('vol',with_process=compute_median, named='med')


worm.segmentation_channel_is('mba30')
worm.extract_time_series_data_for('vol')


gui=Display()
gui.worm=worm


if '/151125_1814_Vol_NaCl_10mM_30s_4min_W2' in data.values():
    gui.worm.center.set_value([13386,93600,36288])#nm
    gui.worm.radius.set_value([18386,26144,37692])#nm


elif '/151217_2109_Vol_NaCl_50mM_30s_4min_W5' in data.values():
    gui.worm.center.set_value([15210, 74445, 60228])
    gui.worm.radius.set_value([19418, 34567, 31525])

elif '/151214_1849_Vol_NaCl_10mM_30s_4min_W1' in data.values():
#    ##FOV1##
#    gui.worm.center.set_value([14186, 98085, 107478])
#    gui.worm.radius.set_value([9593, 23563, 37714])
#    ##FOV2##
#    gui.worm.center.set_value([136950,57750,13600])#nm
#    gui.worm.radius.set_value([57300,51900,36800])#nm
#    ##FOV3##
#    gui.worm.center.set_value([14186,30705,162162])#nm
#    gui.worm.radius.set_value([14111,16112,30490])#nm
#    ##FOV4##
#    gui.worm.center.set_value([14186,36675,221445])#nm
#    gui.worm.radius.set_value([14111,16112,30490])#nm
#    ##FOV5##
#    gui.worm.center.set_value([14186,75990,264978])#nm
#    gui.worm.radius.set_value([14111,33532,30490])#nm
#    ##FOV6##
#    gui.worm.center.set_value([14186,117975,304794])#nm
#    gui.worm.radius.set_value([14111,22526,40552])#nm
#    ##FOV7##
#    gui.worm.center.set_value([14186,66495,358533])#nm
#    gui.worm.radius.set_value([14111,47766,39473])#nm
#    ##FOV8###Not motion corrected
#    gui.worm.center.set_value([14186,33270,425250])#nm
#    gui.worm.radius.set_value([14111,18410,39473])#nm
#    ##FOV9###Not motion corrected
#    gui.worm.center.set_value([14186,  75330, 489195])#nm
#    gui.worm.radius.set_value([14111, 37867, 39473])#nm
#    ##FOV10###
#    gui.worm.center.set_value([8512, 109800, 533673])#nm
#    gui.worm.radius.set_value([12020,  9431, 16981])#nm
    ##FOV11###
    gui.worm.center.set_value([16214, 95610, 586278])#nm
    gui.worm.radius.set_value([20038,  22372, 30388])#nm

elif '/151215_2047_Vol_NaCl_10mM_30s_4min_W2' in data.values():
    #FOV1 Head front
    gui.worm.center.set_value([11616,101175,113967])#nm
    gui.worm.radius.set_value([13815,25341,27993])
    
elif '/151214_1930_Vol_NaCl_10mM_30s_4min_W2' in data.values():
    gui.worm.center.set_value([11825,73545,128961])#nm
    gui.worm.radius.set_value([15185,39350,57911])
    
elif '/151217_1915_Vol_NaCl_50mM_30s_4min_W1'in data.values():
    gui.worm.center.set_value([14170,71775,99099])#nm
    gui.worm.radius.set_value([18777,45888,44638])


##Patch1 without neurons
#gui.worm.center.set_value([ 11402, 103425, 344925])
#gui.worm.radius.set_value([9751,28485,29954])


######The order added here determines the left->right order of scenes
gui.add_scene('Whole_Worm', WormScene(downsample=6) )
gui.add_scene('LocalCM', WormScene(is_local_view=True) )
gui.add_scene('Max_Box_Average',WormScene(is_local_view=True) )
#gui.add_scene('Med_Local', WormScene(is_local_view=True) )
gui.add_scene('Time_Series', Worm4DScene(is_local_view=True) )

#gui.add_scene('Before_Stim', WormScene(is_local_view=True) )
#gui.add_scene('During_Stim', WormScene(is_local_view=True) )
#gui.add_scene('After_Stim', WormScene(is_local_view=True) )


#gui.add_scene('Med_Global', WormScene(is_local_view=True) )


gui.hash_volume=dict({
    'Whole_Worm':'cm',
#    'Time_Series':'vol',
    'LocalCM':'cm',
    'Max_Box_Average':'mba30',
#    'Med_Local':'med',
    'Time_Series':'vol',
#    'Before_Stim':'before',
#    'During_Stim':'during',
#    'After_Stim':'after',
    })



gui.start()


#def elementwise_max(arr1,arr2):
#    return np.array([max(x1,x2) for x1,x2 in zip(arr1,arr2)])
#def elementwise_min(arr1,arr2):
#    return np.array([min(x1,x2) for x1,x2 in zip(arr1,arr2)])
##Hack to crop#
#vol=gui.worm['vol']
#pos=vol.pos
#length=vol.length
#zero=np.zeros_like(pos)
#upr=elementwise_min(np.array(vol.X.shape[1:]),pos+length+1)
#lwr=elementwise_max(zero,pos)        
##        I=np.array(vol.X[:,pos[0]:pos[0]+length[0],pos[1]:pos[1]+length[1],pos[2]:pos[2]+length[2],np.newaxis])
#I=np.array(vol.X[:,lwr[0]:upr[0],lwr[1]:upr[1],lwr[2]:upr[2]])
#from skimage import io
#io.imsave(r'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/FOV9/NoCorrectionNeeded_FOV9.tif',I)




stophere



from MotionCorrection import MotionCorrection
mc=MotionCorrection(gui.worm.time_series_data,gui.worm.results_dir,gui.worm.cache_dir)
mc.correct_current_region()





















##This can be used to get the offsets
##It is used internally by HMM.correct
#E=mc.HMM.estimate(mc.dataset,mc.data_folder+'/EstimatedDispl'+'.sima')
#E=mc.HMM.estimate(mc.dataset)
#displacements=E[0]



##l=np.array([fill(d) for d in mc.C.reshape(-1,mc.C.shape[-2],mc.C.shape[-1])])
#D=np.squeeze( mc.C.sequences[0].__array__()  )
#from skimage import io
#io.imsave(mc.data_folder+'/Corrected_with_gaps'+'.sima/volume.tif',D.astype(np.uint16))  
#
#c=mc.C[0]





