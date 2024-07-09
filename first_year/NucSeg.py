# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 09:37:28 2014

@author: melocaladmin
"""
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
io.use_plugin('tifffile')

from skimage.filter import threshold_otsu, threshold_adaptive, rank
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.morphology import disk, watershed
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance as dist
import scipy.cluster.hierarchy as hier

DataFolder="C:\Users\melocaladmin\My Documents\WormPics"
filename="\\20140908 Confocal ZIM294 Step Size Stacks\\14-09-08 Confocal ZIM294 L4 W2 H 1.5um Stack.tif"

image_stack=io.imread(DataFolder+filename)
z_size,x_size,y_size=image_stack.shape



xy_scale=0.5
z_scale=1.5#exp with later

nrows=np.int(np.ceil(np.sqrt(z_size)))
ncols=nrows

fig, axes = plt.subplots(nrows,ncols,figsize=(3*ncols,3*nrows))
for n in range(z_size):
    j=n % ncols        
    i=j // ncols
    axes[i,j].imshow(image_stack[n,...],interpolation='nearest',cmap='gray')

fig.show()    
#for ax in axes.ravel():
#    if not(len(ax.images)):
#        fig.delaxes(ax)
#fig.tight_layout()