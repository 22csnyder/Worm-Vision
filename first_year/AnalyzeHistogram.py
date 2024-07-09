# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 10:02:25 2014

@author: Chris Snyder
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pims
#import matplotlib.gridspec as gridspec
#from WormFuns import stackplot, adjust
#from WormFuns import *
#from sklearn.cluster import KMeans
from WormFuns import stackplot

def reorient(frame):
    return frame[::-1,...]

DataFolder="C:\Users\melocaladmin\My Documents\WormPics"
filename="\\20140908 Confocal ZIM294 Step Size Stacks\\14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack.tif"
data=DataFolder+filename

v=pims.TiffStack(data,process_func=reorient)

#stackplot(v)

L=[]
for frame in v:
#    L.append(frame[::-1,...])#to make order same as imageJ
    L.append(frame)
Data16u=np.asarray(L)
Data8u=Data16u.copy()
Data8u[Data16u>255]=255
Data8u=Data8u.astype(np.uint8,copy=False)
#
X=Data16u[7]
#
x=X.copy()
y=X.copy()
nrows,ncols=X.shape
for i,z in enumerate(x):
    x[i]=int(i) * np.ones([1,ncols]).astype("int")
    if i<ncols:
        for j in range(0,nrows):
            y[j,i]=int(i)

Xf=X.ravel()
x=x.ravel()
y=y.ravel()

#figH=plt.figure()
#ax=figH.add_axes([0,0,40000,1e5])
#ax.hist(Xf,bins=100)
#ax.set_title('uint16 histogram')
#figH.show()
#

plt.figure(0)
plt.hist(Xf,bins=100)
plt.ylim([0,1e5])
plt.title("uin16")
plt.show(block=False)

plt.figure(1)
plt.imshow(Data8u[7],cmap=plt.get_cmap('Greens'))

xidx=x[(1500>=Xf)]
yidx=y[(1500>=Xf)]
plt.figure(2)
#plt.imshow(Data8u[7],cmap=plt.get_cmap('Greens'))
plt.scatter(yidx,xidx,c='k',marker="2")


xidx=x[(1500<Xf) & (Xf<2650)]
yidx=y[(1500<Xf) & (Xf<2650)]
plt.figure(2)
#plt.imshow(Data8u[7],cmap=plt.get_cmap('Greens'))
plt.scatter(yidx,xidx,c='y',marker="2")

xidx=x[(Xf>=2650)]
yidx=y[(Xf>=2650)]
plt.figure(2)
#plt.imshow(Data8u[7],cmap=plt.get_cmap('Greens'))
plt.scatter(yidx,xidx,c='m',marker="2")


plt.show(block=False)


#xdots=
#fig=plt.figure()
#ax=fig.add_axes()
#ax.imshow(Data8u[7])
#ax.scatter



