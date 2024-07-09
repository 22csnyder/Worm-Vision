# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:20:16 2015

@author: cgs567
"""
#from WormBox.ioworm import ls,return_those_ending_in
#from WormBox.WormFuns import getRegionBdry,contiguous_image_at_grid
#from WormBox.ioworm import getPIMWormData, sort_nicely
#from tifffile import imsave
#import matplotlib.pyplot as plt
#import numpy as np



#from WormBox.BaseClasses import Worm, Worm4D , Environment

#Just need to switch directory.
#directory='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150605_1727_Vol_Img_No_Stimulus_W3'
#directory='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150605_1731_Vol_Img_No_Stimulus_W3-1'
#directory='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150605_1758_Vol_Img_No_Stimulus_W4-1'
#directory='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150605_1824_Vol_Img_No_Stimulus_W5'


#'C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/151002_2126_Vol_Img_Glycerol_3ch_Dev_II_Large_Out_W2/stack series'

#env=Environment(directory)
##Round 1
#hyper=Worm4D('Concatenated.tif',15,env)
#hyper.load_data()
#for i in range(0,hyper.n_worms):
#    worm=hyper.spawn_worm(i)
#    worm.save('stacks/worm'+str(i))






directory='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/151002_2126_Vol_Img_Glycerol_3ch_Dev_II_Large_Out_W2/'
name='151002_2126 Stack.tif'


from WormBox.BaseClasses import tWorm,tEnvironment

#read='C:/Users/cgs567/Documents/Corral/Ki-Confocal-2015/151002_2126_Vol_Img_Glycerol_3ch_Dev_II_Large_Out_W2/TIFF/'
#write='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/151002_2126_Vol_Img_Glycerol_3ch_Dev_II_Large_Out_W2/stack series/'
read='/work/03176/csnyder/Corral/Ki-Confocal-2015/151002_2126_Vol_Img_Glycerol_3ch_Dev_II_Large_Out_W2/TIFF/'
write='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151002_2126_Vol_Img_Glycerol_3ch_Dev_II_Large_Out_W2/stack_series/'

re='sc_151002_2126_Vol_Glycerol_3ch_Dev_II_Large_Out_W2_VOL_Gly_3ch_D_II_LO_W2_L4_O20X_F630x150_P11008x500S194_*****.tiff'


worm=tWorm(tEnvironment(read,write))
worm.dask_read_data(re,n_slices=15)#doesn't seem to work on windows


for t in range(108,len(worm.daX)):
    temp_worm=worm[t]
    temp_worm.save('worm-'+str(t))








#failed_at=263#only happens if you dont' call .load_data
#
##Round 2
#hyper=Worm4D('Concatenated Stacks.tif 2806-5400 kept stack.tif',15,env)
#for i in range(0,hyper.n_worms):
#    i_inc=failed_at+i
#    worm=hyper.spawn_worm(i)
#    worm.save('Stacks/worm'+str(i_inc))




#####Error Message to Dan ####
#
#import PIL,tifffile
#import numpy as np
#import pims
#import skimage.io as io
#
##Two different ways to read in the data:
#full_file='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/150603_1703_Vol_Img_No_Stimulus_W2-2/Concatenated Stacks.tif'
#v=pims.TiffStack(full_file,process_func=lambda frame:np.rot90(frame,3))
#A=io.imread(full_file)
#
#print len(v) #5400
#print A.shape#(5400L, 300L, 1260L)
#
#j=2815
#x=v[j]#OKAY! for j<2816
#j=2816
#x=v[j]#FAILS with PIL\ImageFile.py, line169 OverflowError (shown above) for j>=2816
#x=A[j]#OKAY for all j
#
#
##It's not because those frames are bad
##Load data cropped with imagej to frames 2816-5400
#short='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/150603_1703_Vol_Img_No_Stimulus_W2-2/Concatenated Stacks.tif 2815-5400 kept stack.tif'
#s=pims.TiffStack(short,process_func=lambda frame:np.rot90(frame,3))
#
#j=20
#x=s[20]#OKAY for all j
#
##double check this is the data in question
#print np.all(s[0]==A[2816])#TRUE
#
##Python2.7
#print PIL.VERSION# 1.1.7
#print pims.__version__# 0.2.2
#print tifffile.__version__# 0.5









#idx=2805
#w=[v[i] for i in range(idx,idx+15)]
#X=np.vstack([w_[np.newaxis,:] for w_ in w])
#
#for i in range(idx,idx+15):
#    print i
#    print v[i]
    
    
    
#io.imsave(path+'/pim30-44_rotate.tiff',X)


#hyperstack='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/150603_1703_Vol_Img_No_Stimulus_W2-2/HyperStack.tif'
#
##also works:
##wormMovie=Worm4D(concatenated,15)



#worm=hyper.spawn_worm(2)
#worm.save('pls')
#



#wormMovie=Worm4D(rel_name,15,env)




####This works beautifully without inverting anything ###
##c,z,x,y
#X=io.imread(hyperstack)#Can read hyperstack
#im=X[110,2,:,:]
#io.imsave(path+'/out110.2.tiff',im)
#########################################################
#
#
#stack=concatenated
#
##pims can't read hyperstacks
##reads in as c*z,x,y





#This works:
#import pims
#w=[v[i] for i in range(30,45)]
#X=np.vstack([w_[np.newaxis,:] for w_ in w])
#io.imsave(path+'/pim30-44_rotate.tiff',X)




#io.imsave(path+'/pim220_no reorrient.tiff',v[220])

#imsave(WormConfig['working directory']+'/'+'msc.tif',dst,compress=1)
#
##%%
#Slices=np.array(slice_list)
#np.save(WormConfig['working directory']+'/'+'Slices.npy',Slices)
