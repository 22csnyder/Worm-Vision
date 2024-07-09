# -*- coding: utf-8 -*-
"""
Created on Thu Jan 08 09:07:26 2015

@author: melocaladmin
"""
import numpy as np
import math
#import cv2

def clean8(img16u):
    X8=fiji8convert(img16u,10,99)
    return cv2.medianBlur(X8,5)

def smartcv8convert(img16u,verbose=0):
    X=np.copy(img16u)
    h=np.histogram(img16u,bins=100)
    i=0
    cumul=0
    while i<h[0].shape[0]:
        if h[0][i]<h[0][i+1]:
            break
        i=i+1
        cumul=cumul+h[0][i]
    bottom=h[1][i]
    
    if verbose==1:
        print("bottom is", bottom)
        print("number pixels zeroed is",cumul)
    
    X[X<bottom]=0
    X[X>=bottom]=X[X>=bottom]-bottom
    X=X.astype(np.float32,copy=False)
    X*=255/X.max()
    X=X.round()
    X=X.astype(np.uint8,copy=False)
    return X

def adjust(v,b,c):#brightness and contrast
    c=c*math.pi/4#range from 0 to pi/4
    #slope intercept
    slope=math.tan(c)
    yint=0.5-b*slope
    for frame in v:
        frame=frame*slope+yint
        frame[frame<0]=0
        frame[frame>1]=1
    return v   

def scaleimagetolimits(src,lowest,highest):
    highest=min(src.max(),highest)-lowest
    lowest=max(lowest,src.min())
    dst=(src.copy()-lowest)/highest
    dst[dst<0]=0
    dst[dst>1]=1
    return dst

def setimagelimits(src,lowest,highest):
    dst=src.copy()
    dst=np.copy(src)
    dst[dst<lowest]=lowest
    dst[dst>highest]=highest
    dst-=lowest
    return dst

def autoFijiStyleScaling(src,low_percent=10,high_percent=90):
#    dst=np.copy(src)
    
#    h=np.histogram(src,bins=100)
#    totalpixels=src.size
#    lwrbd=round(low/100.0*totalpixels)
#    uprbd=round(upp/100.0*totalpixels)
#    cumul=0;i=0
#    while 1:
#        cumul+=h[0][i]
#        if cumul>lwrbd:
#            low=h[1][i]
#            lwrbd=totalpixels+1
#        if cumul>uprbd:
#            high=h[1][i+1]
#            break
#        i+=1
#        if i>100:
#            print("something went wrong in autoFijiStyleScaling")
#            break
#    dst=setimagelimits(src,low,high)
    low=np.percentile(src.ravel(),low_percent)
    high=np.percentile(src.ravel(),high_percent)
    return scaleimagetolimits(src,low,high)
#    print "low is",low
#    print "high is",high
#    return dst
        
def shrink_as_is_to_uint8(src):
    dst=np.copy(src)
    dst=dst.astype(np.float32,copy=False)
    dst*=255/dst.max()
    dst.round()
    dst=dst.astype(np.uint8,copy=False)
    return dst
    
def fiji8convert(src,lower_percentile=10,upper_percentile=90):
    return shrink_as_is_to_uint8(autoFijiStyleScaling(src,low=lower_percentile,upp=upper_percentile))