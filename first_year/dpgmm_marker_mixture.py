# -*- coding: utf-8 -*-
"""
Created on Sun May 24 05:01:57 2015

@author: cgs567
"""


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


my_marker_no=[68,72,80,81,85,89,102,103,112,118,120,143,163]

Slices=np.load(WormConfig['working directory']+'/'+'Slices.npy')

#Pathological
my_marker_no=[68,72,80,81,85,89,102,103,112,118,120,143,163]
path_slices=[Slices[n] for n in my_marker_no]
pathological_slices=np.hstack(path_slices)
imsave(WormConfig['working directory']+'/'+'pathological_slices.tif',pathological_slices,compress=1)
#Typical
my_marker_no=[1,67,90,93,112,140,145,171]
typical_slices=[Slices[n] for n in my_marker_no]
typical_slices=np.hstack(typical_slices)
imsave(WormConfig['working directory']+'/'+'typical_slices.tif',typical_slices,compress=1)

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
img=Slices[16]
img=img.astype(np.float)
img/=img.mean()#eh
idx,idy=np.indices(img.shape)
pos=np.vstack([idx.flatten(),idy.flatten()]).transpose()
X=np.transpose(np.vstack([mat.flatten() for mat in [idx,idy,img]]))
timg=np.zeros(img.shape)
#%%

#first try without
from sklearn import mixture

clfG=mixture.GMM(n_components=3, covariance_type='full', n_iter=1000)
clfG.fit(X)
Yg_ = clfG.predict(X)
g_timg=timg.copy()
for i in np.unique(Yg_):
    g_timg[pos[Yg_ == i, 0], pos[Yg_ == i, 1]]=i+1
plt.figure()
plt.imshow(g_timg)
plt.title('GMM')

#%%
clfD=mixture.DPGMM(n_components=5, covariance_type='spherical', alpha=10,n_iter=1000)
clfD.fit(X)
Yd_ = clfD.predict(X)
d_timg=timg.copy()
for i in np.unique(Yd_):
    d_timg[X[Yd_ == i, 0], X[Yd_ == i, 1]]=i+1
plt.figure()
plt.imshow(d_timg)
plt.title('DPGMM')

#%%

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl


import itertools
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
for i, (clf, title) in enumerate([
        (mixture.GMM(n_components=2, covariance_type='full', n_iter=100),
         "Expectation-maximization"),
        (mixture.DPGMM(n_components=2, covariance_type='full', alpha=0.01,
                       n_iter=100),
         "Dirichlet Process,alpha=0.01")
#        ,(mixture.DPGMM(n_components=2, covariance_type='diag', alpha=100.,
#                       n_iter=100),
#         "Dirichlet Process,alpha=100.")
             ]):

    print i    
    clf.fit(X)
    splot = plt.subplot(3, 1, 1 + i)
    Y_ = clf.predict(X)
    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-6, 4 * np.pi - 6)
    plt.ylim(-5, 5)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())

plt.show()


#px,py=np.where(img==1)
#fea=np.vstack([px,py]).transpose()
#pos=np.where(binary==1)
#fea=np.vstack(pos).transpose()

#px,py=np.indices(img.shape)
#fea=np.vstack([px.ravel(),py.ravel(),img.ravel()])

##data whitening
#from scipy.linalg import fractional_matrix_power
#u=np.mean(fea,axis=1).reshape(3,1)
#fea=fea-u#centering
#Cov=np.dot(fea,fea.transpose())
#rootCov=fractional_matrix_power(Cov,0.5)
#negrootCov=fractional_matrix_power(Cov,-0.5)
#wfea=np.dot(negrootCov,fea)#whitening transform
