# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 14:22:31 2015

@author: christopher
"""
import math
import numpy as np
from WormFuns import contiguous_image_at_grid,lmap
from formatworm import autoFijiStyleScaling
#import cv2

def getSimple2DFeature(patch):
    dim=patch.shape
    if dim[0]==1 and dim.__len__()==3:
        dim=dim[1:]
        patch=patch[0]
    mid=[d//2 for d in dim]#hopefully dim is even    
    if dim.__len__()==2:
        ind=np.ndindex(2,2)
    elif dim.__len__()==3:
        ind=np.ndindex(2,2,2)    
    feature=[]
    for idx in ind:
        start=np.multiply(idx,mid)
        loc=[range(s,s+m) for s,m in zip(start,mid)]
        feature.append(np.sum(contiguous_image_at_grid(patch,loc)))
    return feature

def getSumIntensity(patch):
    return np.sum(patch)

#def createContextFeatures(local_features,where_features):
    
    
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    
    # quantizing binvalues in (0...16)
    bin_n=16
    bins = np.int32(bin_n*ang/(2*np.pi))
    
    
    h,w=img.shape[-2:]
    hh,ww=[int(h/2),int(w/2)]
    # Divide to 4 sub-squares
    bin_cells = bins[:hh,:ww], bins[hh:,:ww], bins[:hh,ww:], bins[hh:,ww:]
    mag_cells = mag[:hh,:ww], mag[hh:,:ww], mag[:hh,ww:], mag[hh:,ww:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

def dataRange(d1array):
    return np.max(d1array)-np.min(d1array)
    
def getSegmentFeatures(img,seg_img):
    nseg=len(np.unique(seg_img))
    s=seg_img
    
    where=[lmap(np.mean,np.where(s==i)) for i in range(nseg)]
    how_many=[np.count_nonzero(s==i) for i in range(nseg)]
    diam=[lmap(dataRange,np.where(s==i)) for i in range(nseg)]
    normimg=autoFijiStyleScaling(img)#scale to 0-1
    intensity=[np.mean(normimg[s==i]) for i in range(nseg)]
    
#    std_hm=np.std(how_many)
#    std_where=lmap(np.std,zip(*where))
#    std_diam=lmap(np.std,zip(*diam))
#    std_int=np.std(intensity)

#edit here:
#    phi=list(zip(
#        [w[0]/std_where[0] for w in where],
#        [w[1]/std_where[1] for w in where],
#        [h/std_hm for h in how_many],
#        [d[0]/std_diam[0] for d in diam],
#        [d[1]/std_diam[1] for d in diam],
#        [inte/std_int for inte in intensity]))
        
    phi=list(zip(
        [w[0] for w in where],
        [w[1] for w in where],
        [h for h in how_many],
        [d[0] for d in diam],
        [d[1] for d in diam],
        [inte for inte in intensity]
        ))
    phi=[np.array(p) for p in phi]
    return phi

def computeInnerProducts(fea1,fea2):
    K=np.zeros((len(fea1),len(fea2)))
    for i in range(len(fea1)):
        for j in range(len(fea2)):
            K[i][j]=fea1[i].dot(fea2[j])
    return K

def computeKernel(data1,data2,Kernel):#phi must return np array
    xlen=len(data1)
    x2len=len(data2)
    K=np.zeros((xlen,x2len))
    for i in range(xlen):
        for j in range(x2len):
            K[i][j]=Kernel(data1[i],data2[j])
    return K

    
def computeKernelfromPhi(data1,data2,phi):#phi must return np array
    xlen=len(data1)
    x2len=len(data2)
    K=np.zeros((xlen,x2len))
    for i in range(xlen):
        for j in range(x2len):
            K[i][j]=phi(data1[i]).dot(phi(data2[j]))
    return K
    

def gaussianKernel(x,y,sigma):
    z=x-y
    return math.exp(-1*np.linalg.norm( z.dot(1/sigma)  ))    
    
def relativeGaussianKernel(x,y,sigma):
    nan=[np.where(x+y==0)]
    x=x.astype(np.float)
    x[nan]+=.0001#entry of z comes out as 2
    z=2*(x-y)/(x+y)#hopefully no divide by zero
    return math.exp(-1*np.linalg.norm( z.dot(1/sigma)  ))
    
def mixtureGaussianKernel(x1,y1,sigma1,x2,y2,sigma2):
    return gaussianKernel(x1,y1,sigma1)*relativeGaussianKernel(x2,y2,sigma2)
    