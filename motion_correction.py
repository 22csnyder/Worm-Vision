# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 07:55:12 2015

@author: csnyder
"""



from WormBox.BaseClasses import tWorm as Worm
from skimage import io
import numpy as np
import sima



import numpy as np
from exoplanet import sigma_filter
#verbose=True
verbose=False
min_neighbors=3
neighbor_box=5

def fill(na):
    obj_tmp = na
    if obj_tmp.ndim==2:
        nan_indices = np.where(np.isnan(obj_tmp))
        nan_map = np.zeros_like(obj_tmp)
        nan_map[nan_indices] = 1
        nnanpix = int(np.sum(nan_map))
        if verbose == True:
            msg = 'In frame there are {} nan pixels to be corrected.'
            print msg.format(nnanpix)
        #Correct nan with iterative sigma filter
        obj_tmp = sigma_filter(obj_tmp, nan_map, 
                                   neighbor_box=neighbor_box, 
                                   min_neighbors=min_neighbors, 
                                   verbose=verbose) 
        if verbose == True:
            print 'All nan pixels are corrected.'
    return obj_tmp
####Example of use:
#l=[fill(nax) for nax in _nanI]##takes LONG
#_M=np.vstack([s[np.newaxis,:] for s in l])
###Beware: "astype np.uint16" converts nan to 0!
##anI=_nanI.astype(np.uint16)
##_M=np.vstack([fill(nax)[np.newaxis,:] for nax in anI])
#M=_M.reshape(T,Z,Y,X)
#
#
#io.imsave(save_folder+'corrected - gaps filled exoplanet.tif',M.astype(np.uint16))
#
#print 'done'




#folder='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151002_2126_Vol_Img_Glycerol_3ch_Dev_II_Large_Out_W2/'
#'/work/03176/csnyder/Corral/Ki-Confocal-2015/151002_2126_Vol_Img_Glycerol_3ch_Dev_II_Large_Out_W2/TIFF'


#HMM=sima.motion.hmm.HiddenMarkov2D(max_displacement=np.array([2,40,60]))
#HMM=sima.motion.hmm.HiddenMarkov2D()#produced nan in row gaps
#HMM=sima.motion.hmm.HiddenMarkov2D(granularity='frame')
#HMM=sima.motion.hmm.HiddenMarkov2D(granularity=('row',44))


#HMM=sima.motion.hmm.HiddenMarkov3D(granularity='frame',n_processes=16,max_displacement=np.array([2,40,60]))
#HMM=sima.motion.hmm.HiddenMarkov3D(n_processes=8,max_displacement=np.array([2,40,60]))


#HMM=sima.motion.hmm.HiddenMarkov3D(granularity=('column',5),n_processes=16,max_displacement=np.array([2,80,80]))


#HMM=sima.motion.hmm.HiddenMarkov3D(n_processes=16)
#HMM=sima.motion.hmm.HiddenMarkov3D()
#HMM=sima.motion.hmm.HiddenMarkov3D(granularity=('row',8))
#HMM=sima.motion.hmm.HiddenMarkov2D(max_displacement=np.array([40,60]))#produced nan in row gaps


#%%
'''
### So This Section Seems to Work Great ###
HMM=sima.motion.hmm.HiddenMarkov3D(n_processes=8,max_displacement=np.array([2,40,60]))
folder='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151001_2012_Vol_Img_Glycerol_3ch_Dev_II_W3/'
try:
    dataset=sima.ImagingDataset.load(folder+'temp3.sima')
except:
    data='first 60 stacks.tif'
    sub_folder='stack_series/'
    I=io.imread(folder+data)
    x,y,w,h=30,190,120,88
    #X=I[:,7,y:y+h,x:x+w]
    #X=X[:,np.newaxis,:,:,np.newaxis]#(num_frames, num_planes, num_rows, num_columns, num_channels)
    X=I[:,:,y:y+h,x:x+w]
    X=X[:,:,:,:,np.newaxis]#(num_frames, num_planes, num_rows, num_columns, num_channels)
    s=sima.Sequence.create('ndarray',X)
    dataset=sima.ImagingDataset( [s],folder+'Temp.sima')

#C=HMM.correct([s],folder+'corrected4.sima')
C=HMM.correct(dataset,folder+'corrected4.sima')
D=C.sequences[0].__array__()#try np.array(sequence) next time
D2=np.squeeze(D)
D3=np.nan_to_num(D2)
io.imsave(folder+'corrected7.tif',D3.astype(np.uint16))
'''
#%%

#seq=sima.Sequence.create('TIFF',[[folder+'/stack_series/worm-*.tif']])
#seq=sima.Sequence.create('TIFFs',[[folder+'/stack_series/worm-*.tif']])
#read_CM='/work/03176/csnyder/Corral/Ki-Confocal-2015/151002_CM_Glycerol_3ch_Dev_II_Large_Out_W2/TIFF'

#'''


if __name__== '__main__':
    folder='/work/03176/csnyder/Corral/Ki-Confocal-2015/151002_2126_Vol_Img_Glycerol_3ch_Dev_II_Large_Out_W2/'
    save_folder='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151002_2126_Vol_Img_Glycerol_3ch_Dev_II_Large_Out_W2/'
    sub_folder='stack_series/'
    ######## "No finite observation probabilities"#########
    #HMM=sima.motion.hmm.HiddenMarkov2D()#produced nan in row gaps
    
    data_file='Exp x14 y141 w100 h100 ti30 tf90 zi5 zf10.sima'
    try:
        dataset=sima.ImagingDataset.load(save_folder+data_file)
    except:
        from WormBox.BaseClasses import tWorm
        read_dir=folder+'TIFF/'
        re='sc_151002_2126_Vol_Glycerol_3ch_Dev_II_Large_Out_W2_VOL_Gly_3ch_D_II_LO_W2_L4_O20X_F630x150_P11008x500S194_*****.tiff'
        worm=tWorm(read_dir)
        worm.dask_read_data(re,n_slices=15)
        ti,tf,zi,zf=[30,90,5,10]
        x,y,w,h=[14,141,100,100]
        
    #    X=worm.daX[80:100,7,y:y+h,x:x+w].compute()
        X=worm.daX[ti:tf,zi:zf,y:y+h,x:x+w].compute()
    #    Xc=X[:,np.newaxis,:,:,np.newaxis]
        Xc=X[:,:,:,:,np.newaxis]
        seq=sima.Sequence.create('ndarray',Xc)
        dataset=sima.ImagingDataset( [seq],save_folder+data_file)
        
        
    #C=HMM.correct([seq],save_folder+'corrected.sima')
        
    try:
        C=sima.ImagingDataset.load(save_folder+'corrected.sima')
    except:
        HMM=sima.motion.hmm.HiddenMarkov3D()
        C=HMM.correct(dataset,save_folder+'corrected.sima')
    #Correct=np.squeeze(C.sequences[0])
    #io.imsave(save_folder+'corrected.tif',Correct.astype(np.uint16))#Looks unchanged relative to normal#
    #U=np.squeeze(np.array(dataset.sequences[0])).astype(np.uint16)
    #io.imsave(save_folder+'uncorrected.tif',X)
    
    #'''
    
    
    
    #from sima.sequence import _fill_gaps
    #nG=_fill_gaps(iter(C.sequences[0]),iter(C.sequences[0]))
    #N=np.squeeze(np.vstack(list(nG)))
    #io.imsave(save_folder+'corrected - no gaps.tif',N.astype(np.uint16))
    
    #mat=np.array(C.sequences[0])
    #matT=np.swapaxes(mat,2,3)
    #CT=sima.Sequence.create('ndarray',matT)
    #nH=_fill_gaps(iter(CT),iter(CT))
    #T=[np.transpose(c) for c in C.sequences[0]]
    
    
    
    pause_here
    
    
    ####couldn't get this to work####...
    
    
    nanI=np.squeeze(C.sequences[0].__array__())
    T,Z,Y,X=nanI.shape
    _nanI=nanI.reshape(T*Z,Y,X)
    
    
    #from skimage.filters.rank import median
    #from skimage.morphology import disk
    ##_M=np.vstack([median(na,disk(5)) for na in _nanI.astype(np.uint16)])
    #selem=np.array([[1,1,1,1,1],[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1]])
    #_M=np.vstack([median(na,selem) for na in _nanI.astype(np.uint16)])
    #M=_M.reshape(T,Z,Y,X)
    #io.imsave(save_folder+'corrected - median filter.tif',M)
    #N=nanI
    
    #N=np.squeeze(np.vstack(list(nG)))
    
    
    #nan_indicies=np.where(np.isnan(_nanI))
    #nan_map=np.zeros_like(_nanI)
    #nan_map[nan_indicies]=1
    ##mask=1-nan_map
    #mask=nan_map
    #_M=np.vstack([median(na,disk(5),mask=np.ones_like(na)) for na,m in zip(_nanI.astype(np.uint16),mask) ])
    #M=_M.reshape(T,Z,Y,X)
    #io.imsave(save_folder+'median filter.tif',M)



    l=[fill(nax) for nax in _nanI]##takes LONG
    _M=np.vstack([s[np.newaxis,:] for s in l])
    ##Beware: "astype np.uint16" converts nan to 0!
    #anI=_nanI.astype(np.uint16)
    #_M=np.vstack([fill(nax)[np.newaxis,:] for nax in anI])
    M=_M.reshape(T,Z,Y,X)
    
    
    io.imsave(save_folder+'corrected - gaps filled exoplanet.tif',M.astype(np.uint16))
    
    print 'done'







#N[np.isnan(N)]=M[np.isnan(N)]
#io.imsave(save_folder+'corrected - gaps filled with median.tif',N.astype(np.uint16))


#M=median(nanI,disk(5))


#%%

#E=HMM.estimate(dataset)
#C=HMM.correct([s],folder+'corrected.sima')


#C=HMM.correct(dataset,folder+'corrected.sima')
#D=C.sequences[0].__array__()#try np.array(sequence) next time
#D2=np.squeeze(D)
#D3=np.nan_to_num(D2)


#%%

#Correct=np.squeeze(C.sequences[0])
#io.imsave(folder+'corrected.tif',Correct.astype(np.uint16))
#io.imsave(folder+'uncorrected_rotminus30.tif',X)
#    from scipy.ndimage.interpolation import rotate
    #X=rotate(X,-30,(2,1)) # already returns a numpy array from dask argument

    #x,y,w,h=[0,44,325,232]
    #x,y,w,h=[20,53,197,191]

#%%
#io.imsave(folder+'corrected.tif',D3.astype(np.uint16))

#try:
#    dataset=sima.ImagingDataset.load(folder+'Temp.sima')
#except:
#    I=io.imread(folder+data)
#    x,y,w,h=30,190,120,88
#    #X=I[:,7,y:y+h,x:x+w]
#    #X=X[:,np.newaxis,:,:,np.newaxis]#(num_frames, num_planes, num_rows, num_columns, num_channels)
#    X=I[:,:,y:y+h,x:x+w]
#    X=X[:,:,:,:,np.newaxis]#(num_frames, num_planes, num_rows, num_columns, num_channels)
#    s=sima.Sequence.create('ndarray',X)
#    dataset=sima.ImagingDataset( [s],folder+'Temp.sima')


##This call won't work: only picks up first plane because too large
#import skimage.io
#a=io.imread('/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151002_2126_Vol_Img_Glycerol_3ch_Dev_II_Large_Out_W2/stack_series/worm-13.tif')



#try:
#    io.imsave(folder+'uncorrected60.tif',np.squeeze(X))
#except:
#    pass#X not defined here


#
#Dst=[]
#
#for vec,frame in zip(E[0],np.squeeze(X)):
#    x,y=vec
#    
#    Dst.append(np.lib.pad(frame,((x,y),(0,0)),'constant',constant_values=(0,0) )




#X=np.array([[[0,0,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0]]])




#PT=sima.motion.PlaneTranslation2D()
#
#PT.estimate(dataset)




#folder='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151001_2012_Vol_Img_Glycerol_3ch_Dev_II_W3/first 60 stacks/'
#worm=Worm(folder)
#COME BACK TO THIS WHEN RESHAPE IS IMPLEMENTED
#re='sc_151001_2012_Vol_Glycerol_3ch_Dev_II_W3_VOL_Gly_3ch_D_II_W3_L4_O20X_F630x150_P11008x500S192_*****.tiff'
#worm.dask_read_data(re)




#'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151001_2012_Vol_Img_Glycerol_3ch_Dev_II_W3/Vol_Img_Info.ini'




#from movie_analysis import DirectoryResource
#dataset='150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/TIFF/'
#re_file='sc_150902_1703_Vol_Dev_II_No_Stimulus_W1_W1_ZIM294_L4_No_Stimulus_O20X_F630x150_P11008x500S188_*****.tiff'
#io_pal=DirectoryResource(dataset)
#worm=Worm()
#worm.read_dir=io_pal.read_dir
#worm.dask_read_data(re_file,15)



