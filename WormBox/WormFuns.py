# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 09:17:02 2014

@author: Chris Snyder
"""
#import cv2
#import matplotlib.pyplot as plt
import numpy as np
#import cv2
#import pims
import matplotlib.gridspec as gridspec
import math
#from mpl_toolkits.mplot3d import axes3d
import sys
#import numbers
import collections#check if iterable
#import cv2


def npinfo( unk ):
    if isinstance(unk,np.ndarray):
        print 'Var is a numpy array'
        print 'shape: ',unk.shape
    else:
        try:
            s=unk[0].shape
            print 'Var is a list'
            print 'Var[0] has shape ',s
        except:
            print 'nested list of at least 2 layers'


def getFlatIndicies(nrows,ncols):
    x=np.zeros([nrows,ncols])
    y=np.zeros([nrows,ncols])

    for i in range(0,nrows):
        x[i]=int(i) * np.ones([1,ncols]).astype("int")
        if i<ncols:
            for j in range(0,nrows):
                y[j,i]=int(i)
    x=x.ravel()
    y=y.ravel()
    return x,y
    
def tostring(array):    
    L=[]
    for a in array:
        L.append(str(a))
    return L

def image_at_grid(img,indicies):
    if img.shape.__len__()==2:
        out=img[indicies[0],...][...,indicies[1]]
    elif img.shape.__len__()==3:
        out=img[indicies[0],...,...][...,indicies[1],...][...,...,indicies[2]]
    return out
    
def contiguous_image_at_grid(img,indicies):
    #Should have speedup if the indicies are contiguous
    bounds=[[ind[0],ind[-1]] for ind in indicies]
    b=bounds
    if img.shape.__len__()==2:
        return img[b[0][0]:b[0][1],b[1][0]:b[1][1]]
    elif img.shape.__len__()==3:
        return img[b[0][0]:b[0][1],b[1][0]:b[1][1],b[2][0]:b[2][1]]
    elif img.shape.__len__()==1:
        return img[b[0][0]:b[0][1]]

        
#def contiguous_image_at_grid(img,indicies):
    
def alt_image_at_grid(img,indicies):
    print("some effort to speed up using np.take()")
    print("found to be slower than image at grid")
    if img.shape.__len__()==2:
        print("2d speedup not supported yet (image_at_grid)")
        out=img[indicies[0],...][...,indicies[1]]
    elif img.shape.__len__()==3:
        #method uses 1D indexing of 3d array
        flatidx=[rr*(img.shape[1]*img.shape[2])+cc*img.shape[2]+zz for rr in indicies[0] for cc in indicies[1] for zz in indicies[2]]
        sz=[ind.__len__() for ind in indicies]
#        out=img.take(flatidx).reshape(sz)
        out=img.take(flatidx)
    return out   

def getRegionWrap(src,middle,window_width,radius_specified=0):
    dims=src.shape
    if radius_specified==0:
        half_width=[int(0.5*float(w)) for w in window_width]
    indicies=[]
    for d,m,wid in zip(dims,middle,half_width):
        less=(m-wid)%d
        more=(m+wid)%d
        status=[]
        if more>=less:
            indicies.append(list(range(less,more+1)))
            status.append('no rollover')
        if less > more:
            idx1=list(range(less,d))
            idx2=list(range(0,more))
            status.append(idx2.__len__())
            idx1.extend(idx2)
            indicies.append(idx1)
#    return indicies
    return image_at_grid(src,indicies)

def mirrorArray(src):#concatenate with self in every direction
    temp=src[:]
    for ax in range(src.shape.__len__()):
        temp=np.concatenate((temp,temp),axis=ax)
    return temp

def getContiguousRegionWrap(src,middle,window_width,radius_specified=0):
        #should go faster
    if not isinstance(window_width,collections.Iterable):window_width=[window_width]#handles1D case
    if not isinstance(middle,collections.Iterable):middle=[middle]#handles1D case
    dims=src.shape
    if radius_specified==0:
        half_width=[int(0.5*float(w)) for w in window_width]
    indicies=[]
    for d,m,wid in zip(dims,middle,half_width):
        less=(m-wid)%d
        more=(m+wid)%d
        if more>=less:
            indicies.append(list(range(less,more+1)))
    #            print("no rollover")
        if less>more:
            more=more+d
            indicies.append(list(range(less,more+1)))
    #            print(isinstance(less,int))
    #            print(isinstance(more,int))
    mirror_src=mirrorArray(src)
    return contiguous_image_at_grid(mirror_src,indicies)



def getRegion(middle,delz=2,delx=150,dely=75):
    top=max(0,middle[0]-delx)
    bottom=min(4015,middle[0]+delx)+1
    left=max(0,middle[1]-dely)
    right=min(700-1,middle[1]+dely)+1
    ventral=max(0,middle[2]-delz)
    dorsal=min(27,middle[2]+delz)+1
    region=[range(top,bottom),range(left,right),range(ventral,dorsal)]
    
    newidx=[middle[0]-top,middle[1]-left,middle[2]-ventral]
    return region,newidx

def getRegionBdry(middle,window_radii,img_shape):
    drmax,btmax,rtmax=img_shape#for some reason, img_shape is z first while other two are x first
    delx,dely,delz=window_radii
    top=max(0,middle[0]-delx)
    bottom=min(btmax-1,middle[0]+delx)+1
    left=max(0,middle[1]-dely)
    right=min(rtmax-1,middle[1]+dely)+1
    ventral=max(0,middle[2]-delz)
    dorsal=min(drmax-1,middle[2]+delz)+1
    region=[[top,bottom],[left,right],[ventral,dorsal]]#[,) style
    newidx=[middle[0]-top,middle[1]-left,middle[2]-ventral]
    return region, newidx


def subsample(img,sampling):#sampling z,x,y #img and sampling must have same dim
    dim=img.shape
    indicies=[]
    for i,sz in enumerate(dim):
        samp=sampling[i]
        idn=np.linspace(1,sz-(sz%samp),sz//samp).astype("int")-1
        indicies.append(idn)
    return image_at_grid(img,indicies)

def returnGrid(imgshape,sampling):
    indicies=[]
    if not hasattr(sampling,"__iter__"):
        sampling=[sampling for i in imgshape]
    for i,sz in enumerate(imgshape):
        samp=sampling[i]
        idn=np.linspace(1,sz-(sz%samp),sz//samp).astype("int")-1# not perfect behavior
        indicies.append(idn)
#    indicies=[for sa,sz in zip(sampling,imgshape)
    if indicies.__len__()==1:indicies=indicies[0]
    return indicies


def nothing(x):
    pass

def lmap(fun,it):
    return list(map(fun,it))
def lzip(a,b):
    return list(zip(a,b))

def drawCircle(img,center,radius,color,thickness=1):#(row,col) format to reduce confusio
    transpose_center=(center[-1],center[0])
    return cv2.circle(img.copy(),center=transpose_center,radius=radius,color=color,thickness=thickness)
