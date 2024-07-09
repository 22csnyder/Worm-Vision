# -*- coding: utf-8 -*-
"""
Created on Thu Jan 08 07:48:54 2015

@author: melocaladmin
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pims
#import matplotlib.gridspec as gridspec
#from WormFuns import stackplot, adjust
from WormFuns import *
from formatworm import *
from WormPlot import *
from ioworm import *
from sklearn.cluster import KMeans
from my_watershed import WaterApp
import skimage.measure as m #for connected components
#DataFolder="C:\Users\melocaladmin\My Documents\WormPics"
#filename="\\20140908 Confocal ZIM294 Step Size Stacks\\14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack.tif"
#data=DataFolder+filename
#v=pims.TiffStack(data)

v=getPIMWormData()


#    plt.show(block=False)

#stackplot(v)
#
#L=[]
#for frame in v:
#    L.append(frame[::-1,...])#to make order same as imageJ
#Data16u=np.asarray(L)#to make order same as imageJ
#Data8u=Data16u.copy()
#Data8u[Data16u>255]=255
#Data8u=Data8u.astype(np.uint8,copy=False)


img = v[7]


#Do adaptive local thresholding


src=np.copy(img)

nrows,ncols=src.shape
nwindows=10
#bufferfraction=.1#how much else should be considered for the threshold
windowsize=round(nrows/nwindows)
#buffersize=bufferfraction*windowsize
windowlims=[]
for i in range(nwindows):
    windowlims.append(i*windowsize)

#for i,startwindow in enumerate(windowlims):
#    if i==9:
#        endwindow=nrows-1
#    else:
#        endwindow=startwindow+windowsize
#
#    subimage=src[startwindow:endwindow][:]
#    temp=autoFijiStyleScaling(subimage)
#    temp2=shrink_as_is_to_uint8(temp)
#    
#    ret,thresh=cv2.threshold(temp2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    src[startwindow:endwindow][:]=thresh   
med16=cv2.medianBlur(img,5)
X8=fiji8convert(img,10,99)
clean8=cv2.medianBlur(X8,5)
#th0=cv2.adaptiveThreshold(clean8,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,0)
#
#kernel = np.ones((3,3),np.uint8)
#opening = cv2.morphologyEx(th0,cv2.MORPH_OPEN,kernel, iterations = 2)
#
## sure background area
#sure_bg = cv2.dilate(opening,kernel,iterations=3)
## Finding sure foreground area
#dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
#ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
## Finding unknown region
#sure_fg = np.uint8(sure_fg)
#unknown = cv2.subtract(sure_bg,sure_fg)
#
#
#sidebysideplot([med16,clean8,sure_bg,dist_transform,sure_fg,unknown],['med16','clean8','surebg','dist_tr','sure_fg','unknown'])

Xw=WaterApp(clean8)

Xw.run()

CC=m.label(Xw.markers,neighbors=8)



for c in range(CC.max()):
    fname='RecentSegmentation/region'+str(c)+'.txt'
    locale=np.where(CC==c)
    print_indicies_to_file(locale,fname)
    
## Marker labelling
#ret, markers = cv2.connectedComponents(sure_fg)
#
## Add one to all labels so that sure background is not 0, but 1
#markers = markers+1
#
## Now, mark the region of unknown with zero
#markers[unknown==255] = 0
#
#
#markers = cv2.watershed(img,markers)
#img[markers == -1] = [255,0,0]
#
#cv2.show(img)

#plt.show(block=False)#only call once at the end of program. Use draw() in functions