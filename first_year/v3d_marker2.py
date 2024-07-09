# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 13:10:40 2015

@author: christopher
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
#Read marker script
#ll=ls('./../WormData')
#file_name=ll[1]
#file_name='./../WormPics\\141226_CM_Pan_GCamp_Vol1_Stacks'
#file_name='./../WormData\\141226_ZIM294 HiRes_L4_O20X_F630x75um_P11008x500S173 stack __ Markers150211_attempt2'

#ll=ls(WormConfig['WormPics directory']+'/150226_CM_Hi-res_40x/W5/stacks',1,0)
#apofiles=return_those_ending_in(ll,'apo')
#file_name=apofiles[0]
file_name='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal/150226_CM_Hi-res_40x/W5/stacks/150226_ZIM294_L4_W5 neuron_location_chris.apo'
f=open(file_name,'r')

file_extension=None
if file_name.split('.')[-1]=='apo':
    file_extension='.apo'
elif file_name.split('.'[-1])=='.marker':
    file_extension='.marker'

#for line in f:
#    print(line.split(',',-1))
def parse_apo_line(line):
    pieces=line.split(',')
    z=int(pieces[4].strip())
    x=int(pieces[5].strip())
    y=int(pieces[6].strip())
    return [x,y,z]
def parse_marker_line(line):
    if line[0]=='#':pass
    else: return list(map(int,line.split(',',-1)[0:3]))
def parse_line(line):
    if file_extension=='.apo':
        return parse_apo_line(line)
    elif file_extension=='.marker':
        return np.array(parse_marker_line(line))
    else:
        raise Exception('file_extension not .apo or .marker')

#li=[line for line in f if line[0]!='#']
marker_list=[parse_line(line) for line in f if line[0]!='#']#in x, y, z#row,col,z
stack_path='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal/150226_CM_Hi-res_40x/W5/stacks/150226 Hi-res_40x W5.tif'
#'./../Corral/Snyder/WormPics/150226_CM_Hi-res_40x/W5/stacks\\150226 Hi-res_40x W5.tif'

v=skio.imread(stack_path)#only works in python2 #otherwise need MultiImage class
#v=v.transpose((0,2,1))


z_range=[min(list(zip(*marker_list))[2]),max(list(zip(*marker_list))[2])]
slices_per_neuron=10
neuron_window=np.array([75,75,35])#must be odd
window_radii=(neuron_window-1)/2#x,y,z
rel_marker_coord=(neuron_window-1)/2#coordinate of marker inside neuron_window volume
r_row,r_col,r_z=window_radii


neuron_radius=12
neuron_center=(34,34)



#npoints=16 #must be at least 5 for circle
#radius=neuron_radius
#alpha=.1
#beta=.1
#edge_scale=5000.0
#midpoint=np.array([35,35])
#points=[]
#for i in range(npoints):
#    x=midpoint[0]+radius*math.cos(2*math.pi*i/npoints)
#    y=midpoint[1]+radius*math.sin(2*math.pi*i/npoints)
#    points.append(np.array([int(x),int(y)]))
#points=np.array(points)

marker_no=[str(n) for n in range(len(marker_list))]
#my_marker_list=[marker_list[5],marker_list[6]]

my_marker_list=marker_list[:25]
my_marker_no=marker_no[:25]
my_marker_list=marker_list[25:50]
my_marker_no=marker_no[25:50]
my_marker_list=marker_list[50:75]
my_marker_no=marker_no[50:75]
my_marker_list=marker_list[75:100]
my_marker_no=marker_no[75:100]
my_marker_list=marker_list[100:125]
my_marker_no=marker_no[100:125]
my_marker_list=marker_list[125:150]
my_marker_no=marker_no[125:150]
my_marker_list=marker_list[150:175]
my_marker_no=marker_no[150:175]
my_marker_list=marker_list[0:175]
my_marker_no=marker_no[0:175]

neuron_window=np.array([75,75,35])#must be odd
window_radii=(neuron_window-1)/2#x,y,z
def get_middle_slice(m,v=v):
    z=m[2]
    low,high=m-window_radii,m+window_radii+1
    return v[z,low[0]:high[0],low[1]:high[1]].copy()


 


from skimage.draw import circle

def getInteriorCircleDensity(img,center,radius,nbins=10):
    x,y=circle(center[0],center[1],radius)#y first
    hi,bi=np.histogram(img[x,y],bins=nbins)
    hi=hi.astype(np.float)
    hi/=np.sum(hi)
    return hi
    
def getCenters(img,radius,sampling):   
    dims=img.shape-np.array([2*radius,2*radius])
    return [grid+radius for grid in returnGrid(dims,sampling)]

from scipy.stats import entropy
def CalcCircleEntropy(img,center,radius):
    return entropy(getInteriorCircleDensity(img,center,radius))

#%%
##my_marker_no=[72,80,81,89,101,103,112,120,143]
#my_marker_no=[68,72,80,81,85,89,102,103,112,118,120,143,163]
#
#my_marker_no=list(range(len(marker_list)))#all of them
##my_marker_no=[81,101]
#
#
#my_marker_list=[marker_list[n] for n in my_marker_no]
#
#
#slice_list=[get_middle_slice(m) for m in my_marker_list]
#
##Weird bug. Must copy array before drawing on it.
#draw_list=[cv2.circle(s.copy(),neuron_center,neuron_radius,1.5*s.max(),1) for s in slice_list]
#squarebysideplot(draw_list,map(str,my_marker_no))
#
#dst=np.hstack([d for d in draw_list])
#
#
#imsave(WormConfig['working directory']+'/'+'badeggs.tif',dst,compress=1)

#%%

#img=slice_list[7]
#
#idx,idy=getCenters(img,neuron_radius,(1,1))
#
#Hentropy=np.zeros(img.shape)
#
#for x in idx:
#    for y in idy:
#        Hentropy[x,y]=CalcCircleEntropy(img,(x,y),neuron_radius)
#        
#sidebysideplot([img,Hentropy])

#%%

#Hentropy*=(img.max()/Hentropy.max())
#Hentropy=Hentropy.astype(img.dtype)
#dst=np.hstack([img,Hentropy])
#imsave(WormConfig['working directory']+'/'+'msc.tif',dst,compress=1)

#%%
#Slices=np.array(slice_list)
#np.save(WormConfig['working directory']+'/'+'Slices.npy',Slices)

#%%

f.close()

