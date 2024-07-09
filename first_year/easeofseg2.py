# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 13:44:19 2015
This is a second attempt at segmentation algorithm
@author: christopher
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import felzenszwalb, slic, quickshift,mark_boundaries
from skimage.util import img_as_float
from skimage import io
from ioworm import ls,return_those_ending_in
import skimage
from functools import partial
from WormPlot import sidebysideplot
import skimage.draw as skdraw

def ReadImages(foldername):
    a=ls(foldername)
    data_tif=return_those_ending_in(a,'tif')
    all_imgs=[img_as_float(io.imread(im,'as_grey')) for im in data_tif]
    v=[im/im.max() for im in all_imgs]#scale to 0-1 32bit float
    return v
        
#Define segmentation

####QUICK
qseg = partial(quickshift,sigma=0, kernel_size=3, max_dist=5, ratio=.9,return_tree=True)
###sigma=0 means no smoothing
def seg(img):
    img=skimage.color.gray2rgb(img)
    return qseg(img)


#####SLIC
#sseg = partial(slic,sigma=0,n_segments=3,max_iter=10)
#def seg(img):
#    img=skimage.color.gray2rgb(img)
#    return sseg(img)


####FELZENSZWALB
#seg=partial(felzenszwalb,scale=100,sigma=0,min_size=3)



#Define Image
A=np.zeros((20,40))
B=A.copy()
rr3,cc3=skdraw.circle(5,10,3)
rr5,cc5=skdraw.circle(13,18,5)
A[rr3,cc3]=1
A[rr5,cc5]=1
B[rr3,cc3+12]=1
B[rr5,cc5+12]=1
v=[A,B]


path='./../WormPics/Synthesized/MegaplusTestData'
#path='./../WormPics/Synthesized/2DtestData'
v=ReadImages(path)
s=seg(v[0])

#segments=[seg(img) for img in v]
#s=segments[0]
#s=seg(v[0])

#plt.imshow(s)

#sidebysideplot(s,colormap=-1)

#sidebysideplot([mark_boundaries(img,s) for img,s in zip(X,segments)])



