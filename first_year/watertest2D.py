# -*- coding: utf-8 -*-
"""
Created on Thu Jan 08 07:48:54 2015

@author: melocaladmin
"""

import matplotlib.pyplot as plt
import numpy as np
import pims
#import matplotlib.gridspec as gridspec
#from WormFuns import stackplot, adjust
from WormBox.WormFuns import *
from WormBox.formatworm import *
from WormBox.WormPlot import *
from WormBox.ioworm import ls, pathWormPics
from sklearn.cluster import KMeans
from Watershed.my_watershed import WaterApp
import skimage.measure as m #for connected components
#from opencv2 import cv2#neccesary on linux for multiple version cv2 confusion
import cv2 #cv 3.0.0-beta on winpy2.7


#DataFolder="C:\Users\melocaladmin\My Documents\WormPics"
#filename="\\20140908 Confocal ZIM294 Step Size Stacks\\14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack.tif"
#data=DataFolder+filename
#v=pims.TiffStack(data)

#datafile='/home/christopher/Documents/Ben-Yakar/WormPics/20140908 Confocal ZIM294 Step Size Stacks/14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack.tif'
#v=getPIMWormData(datafile)

#folder='./../WormPics/Synthesized/HiResPharyngeal'
folder=pathWormPics+'/Synthesized/CRFtest'
files=ls(folder,1,0)



img=cv2.imread(files[0],0)#modify each time files are added!!!!


 
#med16=cv2.medianBlur(img,5)
X8=fiji8convert(img,10,99)
clean8=cv2.medianBlur(X8,5)

Xw=WaterApp(clean8)

Xw.run()


from PIL import Image
im=Image.fromarray(Xw.truth)
im.save('truth_2neuron_withborder.tif')#'PPM' for .ppm

img=Xw.truth.copy()

#Remove annoying border
img[img==-1]=1

im2=Image.fromarray(img)
im2.save('truth_2neuron.tif')

labels=[]
images=[]
for c in np.unique(img.flatten()):
    wh=np.where(img==c)
    im=np.zeros(img.shape)
    im[wh]=1
    labels.append(c)
    images.append(im)
    
sidebysideplot(images,labels,horiz=False)

#Find unique colors in mat:
#colors=set( tuple(v) for m2d in Xtru for v in m2d )
#
#for c in colors:
#    howmany=np.where(np.all(Xtru==c,axis=-1))
#    print c,howmany[0].__len__()
    

#out=cv2.imread(files[5],0)
#dout=np.copy(out)
#dout=cv2.cvtColor(dout,cv2.COLOR_GRAY2RGB)
#r,c=out.shape
#for i in range(r):
#    for j in range(c):
#        if np.array_equal(dout[i,j],np.array([15,15,15])):
#            dout[i,j]=np.array([128,0,0])
#        if np.array_equal(dout[i,j],np.array([35,35,35])):
#            dout[i,j]=np.array([0,128,0])



#CC=m.label(Xw.markers,neighbors=8)
#
#
#
#for c in range(CC.max()):
#    fname='RecentSegmentation/region'+str(c)+'.txt'
#    locale=np.where(CC==c)
#    print_indicies_to_file(locale,fname)
    
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