# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 21:17:20 2016

@author: csnyder
"""

####Analyzer####
#analyzer=gui.analyzer
#analyzer.calculate_all_intensities_from_segments()
#analyzer.save_intensity(results_dir+'Intensity.pkl')





import sima
from skimage import io
from mayavi import mlab
import numpy as np

from WormBox.ioworm import smakedirs
from motion_correction import fill
import os

def elementwise_max(arr1,arr2):
    return np.array([max(x1,x2) for x1,x2 in zip(arr1,arr2)])
def elementwise_min(arr1,arr2):
    return np.array([min(x1,x2) for x1,x2 in zip(arr1,arr2)])
def arr2fname(arr):
    return '_'.join(map(str,arr))

class MotionCorrection:
    cache_dir=''
    results_dir=''
    
    regions=[]
    def __init__(self,vol,results_dir,cache_dir=None):
        self.vol=vol
        self.results_dir=results_dir
        if cache_dir is not None:
            self.cache_dir=cache_dir
        else:
            self.cache_dir=self.results_dir

    #some read only properties
    @property
    def data_name(self):
        f=arr2fname
        return 'pos'+f(self.vol.pos)+'_len'+f(self.vol.length)
    @property
    def data_folder(self):
        self._data_folder=self.results_dir+'/'+self.data_name
        smakedirs(self._data_folder) #create it if it wasn't already there
        return self._data_folder


    _I=None
    @property
    def I(self):
        return self._I
    @I.setter
    def I(self,value):
        self._I=value
        print 'caching results region:dask->numpy'
        np.save(self.data_folder+'I.npy',self._I)

    def correct_current_region(self):
        print 'converting to np array'
        pos=self.vol.pos
        length=self.vol.length
        zero=np.zeros_like(pos)
        upr=elementwise_min(np.array(self.vol.X.shape[1:]),pos+length+1)
        lwr=elementwise_max(zero,pos)        
        
        self.upr=upr;self.lwr=lwr
#        self.I=np.array(self.vol.X[:,pos[0]:pos[0]+length[0],pos[1]:pos[1]+length[1],pos[2]:pos[2]+length[2],np.newaxis])
        self.I=np.array(self.vol.X[:,lwr[0]:upr[0],lwr[1]:upr[1],lwr[2]:upr[2],np.newaxis])
        
        s=sima.Sequence.create('ndarray',self.I)
        
        #HMM=sima.motion.hmm.HiddenMarkov3D( n_processes=8,max_displacement=np.array([5,20,30]) )
#        HMM=sima.motion.hmm.HiddenMarkov3D(n_processes=16,max_displacement=np.array([2,10,10]))
        self.HMM=sima.motion.hmm.HiddenMarkov3D(granularity='frame',n_processes=16,max_displacement=np.array([1,20,20]))
#        self.HMM=sima.motion.hmm.HiddenMarkov3D(granularity='frame',n_processes=16,max_displacement=np.array([2,70,100]))
#        self.HMM=sima.motion.hmm.HiddenMarkov3D(granularity=('row',8),n_processes=8,max_displacement=np.array([2,20,20]))

###look at using:
#import shutil
#shutil.rmtree('foldername')

        if os.path.isdir(self.data_folder+'/dataset.sima'):
            os.rmdir(self.data_folder+'/dataset.sima')
        dataset=sima.ImagingDataset( [s],self.data_folder+'/dataset.sima')
        self.dataset=dataset
        
        print 'correcting motion...'
        C=self.HMM.correct(dataset,self.data_folder+'/Corrected'+'.sima')
        #C=sima.ImagingDataset.load(folder+'/Corrected_with_gaps'+region+'.sima')
        self.C=C
        
        D=np.squeeze( C.sequences[0].__array__()  )

        io.imsave(self.data_folder+'/Corrected'+self.data_name+'.tif',D.astype(np.uint16))
        print '...saved as: ',self.data_folder+'/Corrected'+self.data_name+'.tif'
#        l=np.array([fill(d) for d in D.reshape(-1,D.shape[-2],D.shape[-1])])
#        X=l.reshape(D.shape)
#
#        self.X=X
#        
#        print 'saving tif results...'
#        io.imsave(self.results_dir+'/motion_correction'+self.data_name+'.tif',X.astype(np.uint16))
#        print '...done'
        



#####################Generate Picture###########################
#scale=gui.worm['cm'].nm_voxel_shape
#from mayavi import mlab
#mlab.figure()
#Arr=gui.worm['cm'].X.compute()[0]
#dar=Arr[:,::5,::5]
#izi,iyi,ixi=np.indices(dar.shape)
#s=mlab.pipeline.scalar_field(izi,5*iyi,5*ixi,dar)
#v=mlab.pipeline.volume(s)
#key,neu=gui.worm.segments.items()[0]
#for key,neu in gui.worm.segments.items():
#    p=neu['marker']//scale
#    mlab.text3d(x=p[0],y=p[1],z=p[2],text=key,orient_to_camera=True,line_width=10,scale=[30,30,30])
#    nodes=mlab.points3d(p[0],p[1],p[2],name='new marker',color=(0,1,0))
#    nodes.glyph.glyph.scale_factor=20
#
#from mayavi import mlab
#Arr=np.mean(I,axis=0)[:,:,:,0]
#mlab.pipeline.volume(mlab.pipeline.scalar_field(Arr))






#np.save(cache_dir+'head_vol.npy',I)


##HMM=sima.motion.hmm.HiddenMarkov3D( n_processes=8,max_displacement=np.array([5,20,30]) )
#HMM=sima.motion.hmm.HiddenMarkov3D(n_processes=16,max_displacement=np.array([2,10,10]))
##HMM=sima.motion.hmm.HiddenMarkov3D(granularity=('row',8),n_processes=8,max_displacement=np.array([2,10,10]))

#region='Head3'
#
#dataset=sima.ImagingDataset( [s],folder+region+'.sima')
#C=HMM.correct(dataset,folder+'Corrected'+region+'.sima')
##C=sima.ImagingDataset.load(folder+'Corrected'+region+'.sima')
#
#D=np.nan_to_num( np.squeeze( C.sequences[0].__array__()  )  )
#
#
#UC=gui.worm['mba30'].X[pos[0]:pos[0]+length[0],pos[1]:pos[1]+length[1],pos[2]:pos[2]+length[2]]
#maxD = np.max(D,axis=0)
#meanD = np.mean(D,axis=0)
#
#
#io.imsave(folder+'corrected'+region+'.tif',D.astype(np.uint16))
#
#D[np.where(D==0)]=np.nan
#from motion_correction import fill
#l=np.array([fill(d) for d in D.reshape(-1,86,83)])
#E=l.reshape(D.shape)
#
#io.imsave(folder+'corrected_and_filled'+region+'.tif',E.astype(np.uint16))


#V=np.std(E,axis=0)
#io.imsave(folder+'std_corr'+region+'.tif',V.astype(np.uint16))
#
#M=np.median(E,axis=0)

'''
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
'''