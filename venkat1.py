# -*- coding: utf-8 -*-
"""
Created on Mon May 11 02:59:43 2015

@author: cgs567
"""

import matplotlib.pyplot as plt
import numpy as np

import WormBox.ioworm as ioworm
import WormBox.WormPlot as myplot
import cv2
import skimage.measure as m
lplot=myplot.squarebysideplot


#    lplot(imglist,list(str(imglist)))

full_file_list=ioworm.ls('C:/Users/cgs567/Documents/Corral/venkat/40X Images for Chris',1,0)
useful_indicies=[3,4,5,9,12,14,32,35]
file_list=[full_file_list[j] for j in useful_indicies]


X=[cv2.imread(f,0) for f in file_list]
Xbin=[cv2.threshold(x,np.median(x),255,cv2.THRESH_BINARY)[1] for x in X]
#Xadth=[cv2.adaptiveThreshold(x,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,0) for x in X]
#kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))    
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
Xopen=[cv2.morphologyEx(x,cv2.MORPH_OPEN,kernel) for x in Xbin]
Xmed=[cv2.medianBlur(x,11) for x in Xopen]

#lplot(Xbin,['Xbin'],'Greys_r')#already saved
#lplot(Xopen,['Xopen'],'Greys_r')#saved
lplot(Xmed,['Xmed'],'Greys_r')
#lplot(Xadth,['Xadth'],'Greys_r')


#Adaptive Histogram Equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
Y=[clahe.apply(x) for x in X]
Ybin=[cv2.threshold(y,np.median(y),255,cv2.THRESH_BINARY)[1] for y in Y]
Yopen=[cv2.morphologyEx(y,cv2.MORPH_OPEN,kernel) for y in Ybin]
lplot(Ybin,['Ybin'],'Greys_r')
lplot(Yopen,['Yopen'],'Greys_r')

#xo=Xopen[3]
#CC_list=[m.label(xo,neighbors=8) for xo in Xopen]
#lplot(CC_list,colormaps='jet')

#lplot(CC_list,colormaps=new_map)


#CC=m.label(xo,neighbors=8)
#RP=m.regionprops(CC)
#ca=[r.convex_area for r in RP]
#
#import matplotlib
#from random import random,seed
#seed(22)
#colors = [(1,1,1)] + [(random(),random(),random()) for i in range(255)]
#new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)
#




##########Take a Label map and turn it into a pretty image and save it#########
#cc=CC_list[3]
##cxo=np.zeros((cc.shape[0],cc.shape[1],3))
#nr,nc=cc.shape;cc_=cc.ravel()
##cxo_=cxo.ravel()
##cxo=cxo_.reshape(nr*nc,3)
#cxo=[]
#for i,c in enumerate(cc_):
#    cxo.append(list((colors[c%len(colors)])))
#    
#cxo=np.array(cxo)
#cxo=cxo.reshape(nr,nc,3)
#cxo*=65536
#cxo=cxo.astype(np.uint16)
#from tifffile import imsave
#imsave('image18 CC.tif',cxo)

#draw=[]
#for xo in Xopen:
#    lines=cv2.HoughLinesP(xo,rho=5,theta=.05,threshold=22,minLineLength=500,maxLineGap=400)[:,0,:]
#    draw_im=cv2.cvtColor(xo,cv2.COLOR_GRAY2RGB)
#    for (x1,y1,x2,y2) in lines[:30]:
#        cv2.line(draw_im,(x1,y1),(x2,y2),(255,0,0),2)
#    draw.append(draw_im)
#myplot.squarebysideplot(draw,colormaps='jet')


#use otsu threshold level to guess canny thresholds
oats=[cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0] for img in X]
Edges=[cv2.Canny(x,0.5*o,o) for x,o in zip(X,oats)]
Contours=[cv2.findContours(e,1,2)[1] for e in Edges]
sketch=[cv2.cvtColor(x,cv2.COLOR_GRAY2RGB) for x in X]
draw=[cv2.drawContours(s,c,-1,(0,255,0),3) for (s,c) in zip(sketch,Contours)]

lplot(draw,['findContours'],colormaps='jet')
#lplot(Edges,['canny'],colormaps='Greys_r')


#(an_img,contours,hierarchy) = cv2.findContours(xo, 1, 2)
#draw_im=cv2.cvtColor(xo,cv2.COLOR_GRAY2RGB)
#draw_im=cv2.drawContours(draw_im,contours,-1,(0,255,0),3)


#lines=cv2.HoughLines(xo,rho=2,theta=0.3,threshold=30)[:,0,:]
#r,p=zip(*lines)
#m,n=X[0].shape
#for (rho, theta) in lines:
#    x0 = np.cos(theta)*rho 
#    y0 = np.sin(theta)*rho
#    pt1 = ( int(x0 + (m+n)*(-np.sin(theta))), int(y0 + (m+n)*np.cos(theta)) )
#    pt2 = ( int(x0 - (m+n)*(-np.sin(theta))), int(y0 - (m+n)*np.cos(theta)) )
#    cv2.line(draw_im, pt1, pt2, (255,0,0), 2) 

#x=cv2.imread(file_list[0],0)
#th=np.median(x)
#ret, thresh = cv2.threshold(x,th,255,cv2.THRESH_BINARY)

#myplot.squarebysideplot(X,colormaps='Greys_r')
#myplot.squarebysideplot(Xbin,colormaps='Greys_r')
#myplot.squarebysideplot(Xopen,colormaps='Greys_r')


#cv2.namedWindow('x',0)
#cv2.imshow('x',x)
#
#cv2.waitKey(0)