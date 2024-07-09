# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 13:50:40 2015

@author: christopher
"""

import numpy as np
from ioworm import ls, sort_nicely
from sklearn.cluster import KMeans
from WormFuns import contiguous_image_at_grid,image_at_grid,subsample,getRegionWrap,returnGrid,getContiguousRegionWrap
import cv2
#from WormPlot import *

from WormPlot import plotClasses

from mlworm import *

    


#######  Read Data    #######
#aaa=ls('../WormPics')
#bbb=ls(aaa[4])
#ccc=ls(bbb[1])
#sort_nicely(ccc)
#img=cv2.imread(ccc[14],0)

img=cv2.imread('../WormPics/20140908 Confocal ZIM294 Step Size Stacks/14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack.tif',0)


#createdataset

if img.ndim==2: img=np.expand_dims(img,axis=0)#view as 3dimg
WindowDim=np.array([1,40,100])#z,x,y
#sampling=[1,50,80]
sampling=[1,20,40]

grid=returnGrid(img.shape,sampling)
grid_dim=[len(g) for g in grid]
keypoints=[[a,b,c] for a in grid[0] for b in grid[1] for c in grid[2]]
getFeature=getSimple2DFeature
#getRegion=getContiguousRegionWrap#Theoretically it will be faster, but have to import mirror() time
getRegion=getRegionWrap
mldata=np.array([getFeature(getRegion(img,pt,WindowDim)) for pt in keypoints])

n_labels=3
#cluster=clu.Kmeans(init='k-means++',n_clusters=n_labels,n_init=5)
cluster=KMeans(init='k-means++',n_clusters=n_labels,n_init=5)
cluster.fit(mldata)


#z_sz,x_sz,y_sz=img.shape
labels=cluster.labels_

plotClasses(labels,keypoints)


#labels=lab.reshape(grid_dim)


#######################



#fig.show()

######
#plotClasses(labels,keypoints)



#sidx=math.floor(roi[2].__len__()/2)
#sidebysideplot(Med8,tostring(roi[2]))
#sidebysideplot(ImageRegion,tostring(roi[2]))
#fig,axes=sidebysideplot(ImageRegion,tostring(roi[2]))
#fig,axes=sidebysideplot(np.zeros(labels.shape))

#plotclasses(labels,subidx,n_labels,fig,axes[sidx])
#
