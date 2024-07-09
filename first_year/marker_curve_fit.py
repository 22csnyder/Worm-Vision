# -*- coding: utf-8 -*-
"""
Created on Sun May 24 08:15:45 2015

@author: cgs567
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 24 05:01:57 2015

@author: cgs567
"""




###This is something I was about to try
###I got a better idea
###Initial attempts here did not work

from WormBox.ioworm import ls,return_those_ending_in
from WormBox.WormFuns import getRegionBdry,contiguous_image_at_grid,returnGrid
from WormBox.ioworm import getPIMWormData, sort_nicely
from WormBox.WormPlot import *
import matplotlib.pyplot as plt
import math
import numpy as np
import skimage.io as skio
import cv2
try:
    from skimage.filters import threshold_otsu,gaussian_filter
except: #is python3
    from skimage.filter import threshold_otsu,gaussian_filter
import skimage.measure as m #for connected components
from tifffile import imsave

pathWormPics='./../Corral/Snyder/WormPics'
WormConfig = {
    'WormPics directory':pathWormPics,
    'working directory':pathWormPics+'/Synthesized/truncatedStitch/150226 Hi-res_40x/W5',
    'data set identifier':'150226 Hi-res_40x W5'
}
#



Slices=np.load(WormConfig['working directory']+'/'+'Slices.npy')


#%%
#img=Slices[18]
img=Slices[16]
img=img.astype(np.float)

toatsu=threshold_otsu(img)
timg=img.copy()
timg[timg<toatsu]=0
timg[timg>=toatsu]=1
pos=np.where(timg==1)
X=np.vstack(pos).transpose()
X=np.vstack([X,X,X])
plt.imshow(timg)

idx,idy=np.indices(img.shape)
pos=np.vstack([idx.flatten(),idy.flatten()]).transpose()

#%%
#Try with intensity
img=Slices[49]
img=img.astype(np.float)
img/=img.mean()#eh
idx,idy=np.indices(img.shape)
pos=np.vstack([idx.flatten(),idy.flatten()]).transpose()
X=np.transpose(np.vstack([mat.flatten() for mat in [idx,idy,img]]))
timg=np.zeros(img.shape)
#%%
def norm(vec):
    return math.sqrt(np.sum(vec**2))
def func(pos,scale,mean,sig):
    pos=np.array(pos)
    mean=np.array(mean)
    sig=np.array(sig)
    normsq=np.sum(((pos-mean)/sig)**2)
    return scale*math.exp( -1*normsq )

import numpy as np
from scipy.optimize import curve_fit

xdata=pos
ydata=img.flatten()

#y = func(xdata, 2.5, 1.3, 0.5)
#ydata = y + 0.2 * np.random.normal(size=len(xdata))
popt, pcov = curve_fit(func, xdata, ydata)




