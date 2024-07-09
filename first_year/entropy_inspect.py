# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 13:47:37 2015

@author: christopher


Conclusion here is that these translation/concatenation based features
are not going to work
Actually they lie on the surface of a n-1 sphere,
so no particular linear classifier is going to "work best"
just by symmetry

"""

from ioworm import ls,return_those_ending_in
import matplotlib.pyplot as plt
import cv2
import numpy as np
#import numbers
#from WormFuns import returnGrid,getContiguousRegionWrap
from WormFuns import *
from functools import partial
import math
from scipy.optimize import minimize


a=ls('./../WormPics/Synthesized/2DtestData')
data_tif=return_those_ending_in(a,'tif')

all_imgs=[cv2.imread(f,0) for f in data_tif]
histbdry=[(np.min(im),np.max(im)) for im in all_imgs]

def getHistFeature(patch,hist_bd,nbins):
    return np.histogram(patch.flatten(),range=hist_bd,bins=nbins)[0]

Xflat=[np.sum(x,axis=1) for x in all_imgs]

w=np.ones(Xflat[0].size)

f=Xflat[0]
hist_bd_flat=(np.min(f),np.max(f))
nbins=10
getFeature=partial(getHistFeature,hist_bd=hist_bd_flat,nbins=nbins)

sampling=[10]

grid=returnGrid(f.shape,sampling)

bdd=histbdry[0]


window_diam=20
patches=[getContiguousRegionWrap(f,elm,window_diam) for elm in grid]
loc_features=[getFeature(_patch,) for _patch in patches]

def getGlobalFeatureAt(i,local_features):#i=1,2,3,...
    n=local_features[0].shape[0]
    return np.concatenate(np.roll(local_features,i*n))
phi=partial(getGlobalFeatureAt,local_features=loc_features)

K=np.zeros((grid.size,grid.size))

for i in range(grid.size):
    for j in range(grid.size):
        K[i][j]=np.dot(phi(i),phi(j))

#normalize each row to sum to 1
#rowsum=np.sum(K,axis=1)
#K/=rowsum[:,np.newaxis]#broadcasting



#alpha=np.ones(grid.size)
#x0=np.ones(grid.size)
def nplog(ary):
    return np.array(list(map(math.log,ary)))

def entropy(a,Ker):
    P=Ker.dot(a)
    P/=P.sum()
    return -1*P.dot( nplog(P) )

H=partial(entropy,Ker=K)

bounds=[(0,None) for i in grid]

t_max=50
t=0
out_dist=np.zeros(grid.size)
while t<t_max:
    x0=np.random.uniform(0,10,grid.size)
    res=minimize(H,x0,method='TNC',bounds=bounds)   
    out_dist+=(res.x/np.max(res.x))
    t+=1    


fig,axes=plt.subplots(3,1)
axes[0].plot(f)
axes[1].plot(out_dist)
axes[2].plot(K.dot(out_dist))




#grid_dim=[len(g) for g in grid]
#keypoints=[[a,b,c] for a in grid[0] for b in grid[1] for c in grid[2]]


#phi=w*f
#
#out=np.convolve(f,phi[::-1])#convolution function does not loop. Bah.
#
#fig,axes=plt.subplots(2,1)
#axes[0].plot(f)
#axes[0].set_title('f')
#axes[1].plot(out)
#axes[1].set_title('conv')



#fig,axes=plt.subplots(4,1)
#for ax,y in zip(axes,Xflat):
#    ax.plot(y)
#fig.canvas.draw()