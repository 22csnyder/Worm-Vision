# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 20:52:05 2014

@author: Chris Snyder
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pims
#import matplotlib.gridspec as gridspec
#from WormFuns import stackplot, adjust
from WormFuns import *
from sklearn.cluster import KMeans
#DataFolder="C:\Users\melocaladmin\My Documents\WormPics"
#filename="\\20140908 Confocal ZIM294 Step Size Stacks\\14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack.tif"
#data=DataFolder+filename
#
#v=pims.TiffStack(data)

v=getPIMWormData()

#stackplot(v)

L=[]
for frame in v:
    L.append(frame[::-1,...])#to make order same as imageJ
Data16u=np.asarray(L)#to make order same as imageJ
Data8u=Data16u.copy()
Data8u[Data16u>255]=255
Data8u=Data8u.astype(np.uint8,copy=False)


#Region of Interest
nucidx=[1083,402,7]#where a nucleus of interest is
roi,roinuc=getRegion(nucidx,delx=200,dely=100,delz=2)#x,y,z format
#Sub8=Data8u[roi[2],...,...][...,roi[0],...][...,...,roi[1]]
Sub8=Data8u

#filter

Filtered8=np.copy(Sub8)
Med8=np.copy(Sub8)
open8=np.copy(Sub8)
close8=np.copy(Sub8)
Dist8=np.copy(Sub8)
MorphGrad8=np.copy(Sub8)
for i,X in enumerate(Sub8):
    medX=cv2.medianBlur(X,7)
    Med8[i]=medX   
    thresh,threshX=cv2.threshold(medX,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    closeX=cv2.morphologyEx(threshX,cv2.MORPH_CLOSE,kernel)
    close8[i]=closeX
    open8[i]=cv2.morphologyEx(threshX,cv2.MORPH_OPEN,kernel)
    Dist8[i]=cv2.distanceTransform(open8[i],cv2.cv.CV_DIST_L2,5)    
    morphgradX=cv2.morphologyEx(closeX,cv2.MORPH_GRADIENT,kernel)
    MorphGrad8[i]=morphgradX



#createdataset
#ImageRegion=Dist8
#s_img,subidx=subsample(ImageRegion,[1,2,2])
#z_sz,x_sz,y_sz=s_img.shape
#mldata=np.zeros([z_sz,x_sz*y_sz])
#for i in range(0,z_sz):
#    mldata[i,:]=s_img[i].reshape(1,x_sz*y_sz)
#
#n_labels=3
#cluster=KMeans(init='k-means++', n_clusters=n_labels, n_init=5)
#cluster.fit(mldata.transpose())
#lab=np.zeros((1,x_sz*y_sz))
#lab[0,:]=cluster.labels_
#labels=lab.reshape(x_sz,y_sz)
#
#sidx=math.floor(roi[2].__len__()/2)
#sidebysideplot(Med8,tostring(roi[2]))
#sidebysideplot(ImageRegion,tostring(roi[2]))
#fig,axes=sidebysideplot(ImageRegion,tostring(roi[2]))
#plotclasses(labels,subidx,n_labels,fig,axes[sidx])

sidebysideplot([Sub8[7],Med8[7],open8[7],close8[7],MorphGrad8[7],2*Dist8[7]],["img","medblur","open","close","grad","Dist Transform"])


    
#Filtered8=np.copy(Sub8)
#Med8=np.copy(Sub8)
#open8=np.copy(Sub8)
#dist8=np.copy(Sub8)
#CircleList=[]
#for i,X in enumerate(Filtered8):
#    medX=cv2.medianBlur(X,7)
#    Med8[i]=medX
#    temp_circ=cv2.HoughCircles(medX,cv2.cv.CV_HOUGH_GRADIENT,dp=1,minDist=25,minRadius=10,maxRadius=30)
#    CircleList.append(temp_circ)    
#    thresh,threshX=cv2.threshold(medX,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#    closeX=cv2.morphologyEx(threshX,cv2.MORPH_CLOSE,kernel)
#    open8[i]=cv2.morphologyEx(threshX,cv2.MORPH_OPEN,kernel)
#    dist8[i]=cv2.distanceTransform(open8[i],cv2.cv.CV_DIST_L2,5)    
#    morphgradX=cv2.morphologyEx(closeX,cv2.MORPH_GRADIENT,kernel)
#    Filtered8[i]=morphgradX
#
#sidebysideplot(dist8,["distance"])
#sidebysideplot(open8,["open"])


##########


    


#fig2,ax2=sidebysideplot(Filtered8,tostring(roi[2]))
#plotdot(roinuc,ax2)
#
#fig,ax=sidebysideplot(Sub8,tostring(roi[2]))
#plotdot(roinuc,ax)



#X=Data8u[12]
#medX=cv2.medianBlur(X,7)
#thresh,threshX=cv2.threshold(medX,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#closeX=cv2.morphologyEx(threshX,cv2.MORPH_CLOSE,kernel)
#morphgradX=cv2.morphologyEx(closeX,cv2.MORPH_GRADIENT,kernel)
#sidebysideplot([X,medX,threshX,closeX,morphgradX],["X","medblur","threshold","close","grad"])


#med8=Data8u
#for frame in med8:
#    frame=cv2.medianBlur(frame,9)
#    frame=cv2.medianBlur(frame,7)


#stackplot(med8)
#stackplot(Data8u)

#sX=med16[:,500:1500,200:500]
#nzidx=np.where(sX>8000)
#scatter3d(nzidx)

#X=Data16[12,:,:]
#
#X=Data8[12,:,:]#just to prototype

#X=Med8[idx];circles=CircleList[idx]
#if circles is not None:
#    circles=np.round(circles[0,:]).astype("int")
#    for (x,y,r) in circles:
#        cv2.circle(X,(x,y),r,(0,255,0),4)
#        cv2.rectangle(X,(x-5,y-5),(x+5,y+5),(0,128,255),-1)
#    cv2.imshow("output",X)
#    cv2.waitKey(0)