# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 13:44:19 2015

@author: christopher
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import lena
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from ioworm import ls,return_those_ending_in
import skimage
from functools import partial
from WormPlot import sidebysideplot

#import cv2
#all_imgs=[img_as_float(io.imread(f,'as_grey')) for f in data_tif[0:2]]
##all_imgs=[img_as_float(io.imread(f,'as_grey')) for f in data_tif]
#X=[im/im.max() for im in all_imgs]
##
##im
#
##img=skimage.color.gray2rgb(io.imread(data_tif[0]))
#img=io.imread(data_tif[0])

#quick requires rgb color image
#seems to take much longer than Slic
#X=[skimage.color.gray2rgb(img) for img in X]
img=skimage.color.gray2rgb(img)
seg = partial(quickshift, kernel_size=3, max_dist=500, ratio=.9,return_tree=True)
#seg=partial(felzenszwalb,scale=1000,sigma=5,min_size=50)
#seg=partial(felzenszwalb,sigma=5,min_size=50)


scales=[10,50,100,500,1000]
sca=1000

segments=[seg(img,scale=sca) for img in X]

#sidebysideplot([mark_boundaries(img,s) for img,s in zip(X,segments)],horiz=True)
#sidebysideplot([mark_boundaries(img,s) for s in segments],horiz=True)



def lmap(fun,it):
    return list(map(fun,it))

#similarity between segments
seg_img=segments[0]

def getSegmentFeatures(img,seg_img):
    nseg=len(np.unique(seg_img))
    #where and how big
    
    #    for i in range(nseg):
    s=seg_img
    how_many=[np.count_nonzero(s==i) for i in range(nseg)]
    where=[lmap(np.mean,np.where(s==i)) for i in range(nseg)]
    diam=[lmap(dataRange,np.where(s==i)) for i in range(nseg)]
    intensity=[np.mean(img[s==i]) for i in range(nseg)]
    
    std_hm=np.std(how_many)
    std_where=lmap(np.std,zip(*where))
    std_diam=lmap(np.std,zip(*diam))
    std_int=np.std(intensity)
    
    phi=list(zip(
        [h/std_hm for h in how_many],
        [w[0]/std_where[0] for w in where],
        [w[1]/std_where[1] for w in where],
        [d[0]/std_diam[0] for d in diam],
        [d[1]/std_diam[1] for d in diam],
        [inte/std_int for inte in intensity]))
    phi=[np.array(p) for p in phi]
    return phi

def lzip(src):
    return list(zip(src))
    
def lunzip(src):
    return list(zip(*src))

def getSegmentMidpoints(seg_img):
    nseg=len(np.unique(seg_img))
    return [lmap(np.mean,np.where(seg_img==i)) for i in range(nseg)]
    
phi0,phi1=[getSegmentFeatures(img,s) for img,s in zip(X,segments)]
FLANN_INDEX_KDTREE=0
#FLANN_INDEX_KDTREE=1
#FLANN_INDEX_LSH=6
index_params= dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#index_params=dict(algorithm=FLANN_INDEX_LSH,table_number=6,
#                  key_size=12,multi_probe_level=1)
MIN_MATCH_COUNT=10
search_params=dict(checks=10)

flann=cv2.FlannBasedMatcher(index_params,{})
matches=flann.knnMatch(np.asarray(phi0,np.float32),np.asarray(phi1,np.float32),k=1)

kp0,kp1=[getSegmentMidpoints(ss) for ss in segments]
kp0,kp1=np.array(kp0),np.array(kp1)
#kp1+=[X[0].shape[0],0]

#kp0,kp1=lunzip(kp0),lunzip(kp1)
#vertically concatenate and draw correpondences:

X_bdd=[mark_boundaries(img,s) for img,s in zip(X,segments)]
view=np.vstack(X_bdd)

#view*=255
#view=view.astype(np.uint8)

w1=X[0].shape[0]
for m in matches:
    m=m[0]
    # draw the keypoints
    # print m.queryIdx, m.trainIdx, m.distance
    color = tuple([np.random.uniform(0, 1) for _ in range(3)])
    cv2.line(view, (int(kp0[m.queryIdx][1]), int(kp0[m.queryIdx][0])) 
    , (int(kp1[m.trainIdx][1]), int(kp1[m.trainIdx][0]) +w1 ), color,cv2.LINE_AA)

plt.imshow(view)
plt.show()

#cv2.imshow("view", view)
#cv2.waitKey()



#distance
#trainIdx#This is the corresponding matched point
#queryIdx
#imgIdx



#K=np.zeros((nseg,nseg))
#for i in range(nseg):
#    for j in range(nseg):
#        K[i][j]=math.exp(phi[i])





#segments=[seg(im) for im in X]
#a,b,c=seg(img)
#s=quickshift(img,kernel_size=3,max_dist=500,ratio=0.9)
#segments=seg(img)   

#slic didn't seem to work. Mainly just rectangular grid
#segments_slic = slic(img, n_segments=300, compactness=1, sigma=[1,3,3],convert2lab=False)

#print("Felzenszwalb's number of segments: %d" % len(np.unique(segments_fz)))
#print("Quickshift number of segments: %d" % len(np.unique(segments_quick)))

#segments=segments_quick
#[print("Number of segments: %d" % len(np.unique(s))) for s in segments]
    
#segments=segments_fz


#fig, ax = plt.subplots(2, 1)
#fig.set_size_inches(2, 8, forward=True)
#fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
#ax[0].imshow(img)
##ax[1].imshow(mark_boundaries(img, segments,outline_color=[1,0,0]))
#ax[1].imshow(mark_boundaries(img, segments))

#sidebysideplot([mark_boundaries(im,s) for im,s in zip(X,segments)],horiz=True)


#ax[1].imshow()

#ax[0].imshow(mark_boundaries(img, segments_fz))
#ax[0].set_title("Felzenszwalbs's method")
#ax[1].imshow(mark_boundaries(img, segments_slic))
#ax[1].set_title("SLIC")
#ax[2].set_title("Quickshift")
#for a in ax:
#    a.set_xticks(())
#    a.set_yticks(())
#plt.show()