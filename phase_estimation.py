# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 13:53:33 2015

@author: christopher
"""

from ioworm import *
import cv2
local_files=ls('./../WormPics/Synthesized/2DtestData')
tif_data=return_those_ending_in(local_files,'tif')
from mlworm import *
from WormPlot import sidebysideplot
    from WormFuns import returnGrid
#v=getPIMWormData(data[0])
#X=cv2.imread(tif_data[0],0)

#Notes for future:
#Alignment looks like it's almost reversed
#Improve feature, see if resolves

all_imgs=[cv2.imread(d,0).transpose() for d in tif_data]
print("image shapes:")
clean_print([x.shape for x in all_imgs])

img=all_imgs[0]

def getPatchFeatures(img):
    nrows,ncols=img.shape
    npatch=24
    idx=np.linspace(0,ncols,npatch+1).astype('int')
    patch_list=[img[idx[i]:idx[i+1],:] for i in range(idx.__len__()-1)]
    features=np.array([getSimple2DFeature(p) for p in patch_list])
    return features

def getSimpleConvolveFeatures(img):
    nrows,ncols=img.shape
    proj=np.sum(img,axis=1)
    nspot=50
    idx=np.linspace(0,nrows,nspot+1)[:-1].astype('int')
    realign=[np.roll(proj,-1*idx[i]) for i in range(nspot)]
    n_samp=2520
    samp_idx=np.linspace(0,ncols,n_samp+1)[:-1].astype('int')
    features=np.array([[r[i] for i in samp_idx] for r in realign])    
    return features,[[i,ncols//2] for i in idx]

#getFeatures=getPatchFeatures
getFeatures=getSimpleConvolveFeatures

features,idx=getFeatures(img)

target=5


def getPhaseWeight(wavelength):#linear
    l=np.linspace(2,-2,wavelength+1)
    wt=[abs(x)-1 for x in l]
    return wt[:-1]
#weight=getPhaseWeight(npatch)
#wt=np.array([weight[np.mod(i-target,npatch)] for i in range(npatch)])
#w=wt.dot(features)

w=features[target]#boring way to do it


dotproduct=[]
location=[]
for img in all_imgs:
    _features,loc=getFeatures(img)
    dotproduct.append([w.dot(v) for v in _features])
#    location.append([[(idx[i]+idx[i+1])//2,ncols//2] for i in range(idx.__len__()-1)])
    location.append(loc)
fig,axes=sidebysideplot(all_imgs)
normalizeddotproduct=[[d/max(dot) for d in dot] for dot in dotproduct]
for ax,loc,val in zip(axes,location,dotproduct):
    y,x=zip(*loc)
#    ax.scatter(x,y,marker='$'+str(val)+'$',s=1000)
    scat=ax.scatter(x,y,c=val,cmap='jet')
fig.colorbar(scat)    
    
x,y=location[0][target][::-1]
axes[0].scatter(x+100,y,marker='*',s=350,c='yellow')
fig.canvas.draw()

