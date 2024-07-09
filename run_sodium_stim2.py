
import numpy as np
#from WormBox.BaseClasses import tEnvironment as Env
from WormBox.VisClasses import MultiChannelWorm
from WormScene import WormScene,Worm4DScene
from Display import Display

#StimOn: step232.8
#StimOff: step291
dataset='151125_1814_Vol_NaCl_10mM_30s_4min_W2'

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

from functools import partial
def during_interval(t0,t1,array):
    return compute_median(array[t0:t1])

def max_box_average(duration,array):
    import dask.array as da
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
before_stim=partial(during_interval,0,t0)
during_stim=partial(during_interval,t0,t1)
after_stim=partial(during_interval,t1,None)

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

######clean generic good starting point
#gui.worm.center.set_value([12608,120975,194607])#nm


##Head Region
#gui.worm.center.set_value([13635,109500,24885])#nm
#gui.worm.radius.set_value([7989,10220,25725])#nm

#MediumHeadRegion for Correction attempt
#gui.worm.center.set_value([11354, 86925, 44856])
#gui.worm.radius.set_value([10086, 14418, 21735])

##Region 3 motion correction
gui.worm.center.set_value([7133, 86925, 42966])
gui.worm.radius.set_value([5498, 14418, 13546])



#Ventral Chord
#gui.worm.center.set_value([13635,121680,468342])#nm
#gui.worm.radius.set_value([7989,10220,25725])#nm

#Neck#Good to use vol4D to illustrate movement
#gui.worm.center.set_value([16611,66870,64512])#nm
#gui.worm.radius.set_value([11330,10220,7989])#nm
#gui.worm.radius.set_value([7989,10220,25725])#nm


#Sample Motion Correction
#'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Cache/corrected7.tif'
#gui.worm.center.set_value([14358,113670,11214])#nm
#gui.worm.radius.set_value([5000,5000,9950])#nm


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




####Analyzer####
#analyzer=gui.analyzer
#analyzer.calculate_all_intensities_from_segments()
#analyzer.save_intensity(save_results_dir+'Intensity.pkl')





import sima
from skimage import io
from mayavi import mlab
folder=cache_dir
vol=gui.worm['vol']
#pos=[0,195,0];length=[19,215,231]#vol.pos;vol.length
#pos=[5,195,0];length=[5,50,50]#vol.pos;vol.length
pos=vol.pos;length=vol.length



pause_here


####Generate Picture####
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


from mayavi import mlab
Arr=np.mean(I,axis=0)[:,:,:,0]
mlab.pipeline.volume(mlab.pipeline.scalar_field(Arr))




I=vol.X[:,pos[0]:pos[0]+length[0],pos[1]:pos[1]+length[1],pos[2]:pos[2]+length[2],np.newaxis].compute()

#np.save(cache_dir+'head_vol.npy',I)


s=sima.Sequence.create('ndarray',I)
#HMM=sima.motion.hmm.HiddenMarkov3D( n_processes=8,max_displacement=np.array([5,20,30]) )
HMM=sima.motion.hmm.HiddenMarkov3D(n_processes=16,max_displacement=np.array([2,10,10]))
#HMM=sima.motion.hmm.HiddenMarkov3D(granularity=('row',8),n_processes=8,max_displacement=np.array([2,10,10]))

region='Head3'

dataset=sima.ImagingDataset( [s],folder+region+'.sima')
C=HMM.correct(dataset,folder+'Corrected'+region+'.sima')
#C=sima.ImagingDataset.load(folder+'Corrected'+region+'.sima')

D=np.nan_to_num( np.squeeze( C.sequences[0].__array__()  )  )


UC=gui.worm['mba30'].X[pos[0]:pos[0]+length[0],pos[1]:pos[1]+length[1],pos[2]:pos[2]+length[2]]
maxD = np.max(D,axis=0)
meanD = np.mean(D,axis=0)


io.imsave(folder+'corrected'+region+'.tif',D.astype(np.uint16))

D[np.where(D==0)]=np.nan
from motion_correction import fill
l=np.array([fill(d) for d in D.reshape(-1,86,83)])
E=l.reshape(D.shape)

io.imsave(folder+'corrected_and_filled'+region+'.tif',E.astype(np.uint16))


#V=np.std(E,axis=0)
#io.imsave(folder+'std_corr'+region+'.tif',V.astype(np.uint16))
#
#M=np.median(E,axis=0)


mlab.pipeline.volume(mlab.pipeline.scalar_field( meanD ))


mlab.figure()
A=meanD
spacing=gui.worm['vol'].spacing
izi,iyi,ixi=np.indices(A.shape)
v=mlab.pipeline.volume(mlab.pipeline.scalar_field(spacing[0]*izi,spacing[1]*iyi,spacing[2]*ixi,A))

mlab.figure()
A=maxD
spacing=gui.worm['vol'].spacing
izi,iyi,ixi=np.indices(A.shape)
v=mlab.pipeline.volume(mlab.pipeline.scalar_field(spacing[0]*izi,spacing[1]*iyi,spacing[2]*ixi,A))

mlab.figure()
A=UC
spacing=gui.worm['vol'].spacing
izi,iyi,ixi=np.indices(A.shape)
v=mlab.pipeline.volume(mlab.pipeline.scalar_field(spacing[0]*izi,spacing[1]*iyi,spacing[2]*ixi,A))

















#spacing=gui.worm['vol'].nm_voxel_shape
#from mayavi import mlab
#mlab.figure()
#Arr=gui.worm['cm'].X.compute()[0]
#dar=Arr[:,::5,::5]
#s=mlab.pipeline.scalar_field(izi,5*iyi,5*ixi,dar)
#v=mlab.pipeline.volume(s)





io.imsave(folder+'corrected7.tif',D3.astype(np.uint16))




X=np.mean(I,axis=0)[0]

C=np.mean(D3,axis=0)
mlab.pipeline.volume(mlab.pipeline.scalar_field(C))
mlab.figure()
#mlab.pipeline.volume(mlab.pipeline.scalar_field(X.astype(np.float)))


###Plot example of motion correction###
MC=io.imread('/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Cache/first motion correct/corrected7.tif')
#UC=gui.worm['mba30'].get_unscaled_padded_patch()
UC=gui.worm['mba30'].X[pos[0]:pos[0]+length[0],pos[1]:pos[1]+length[1],pos[2]:pos[2]+length[2]].compute()
mlab.pipeline.volume(mlab.pipeline.scalar_field(np.mean(MC,axis=0)))
mlab.pipeline.volume(mlab.pipeline.scalar_field(UC))
mlab.pipeline.volume(mlab.pipeline.scalar_field(MC))
first_correct_folder='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Cache/first motion correct/'

flat_mc=np.mean(MC,axis=(0,1))
flat_uc=np.mean(UC,axis=0)

io.imsave(first_correct_folder+'flat_corrected.tif',flat_mc.astype(np.uint16))
io.imsave(first_correct_folder+'flat_uncorrected.tif',flat_uc.astype(np.uint16))



R=io.imread('/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Cache/corrected7_crop_region.tif')
mR=np.mean(R,axis=(1,2,3))
import matplotlib.pyplot as plt
t=np.mgrid[0:len(vol.X)]
plt.plot(t,mR)






import dask.array as da
vol=gui.worm['vol']
pos=vol.pos;length=vol.length
X=vol.X[:,pos[0]:pos[0]+length[0],pos[1]:pos[1]+length[1],pos[2]:pos[2]+length[2]]
daM=da.mean(X, axis=(1,2,3))
patch=daM.compute()
plt.plot(t,patch)
plt.title('Background patch2 Average Intensity decreases with Time')
plt.xlabel('time points')


####Plot average for all frames with time####  (wanted to see if whole image had downward trend
daM=da.mean(vol.X, axis=(1,2,3))

x=daM.compute()

plt.plot(t,x)
plt.title('Average 3D Intensity decreases with Time')
plt.xlabel('time points')

