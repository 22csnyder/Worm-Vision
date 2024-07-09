# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:42:15 2015

@author: cgs567
"""



from WormBox.ioworm import ls,return_those_ending_in
from WormBox.WormFuns import getRegionBdry,contiguous_image_at_grid,returnGrid,drawCircle
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
from sklearn.neighbors import KernelDensity
pathWormPics='./../Corral/Snyder/WormPics'
WormConfig = {
    'WormPics directory':pathWormPics,
    'working directory':pathWormPics+'/Synthesized/truncatedStitch/150226 Hi-res_40x/W5',
    'data set identifier':'150226 Hi-res_40x W5'
}
#

import itertools
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

slices_per_neuron=10
neuron_window=np.array([75,75,35])#must be odd
window_radii=(neuron_window-1)/2#x,y,z
rel_marker_coord=(neuron_window-1)/2#coordinate of marker inside neuron_window volume
r_row,r_col,r_z=window_radii


neuron_radius=12
neuron_center=window_radii[:2]




my_marker_no=[68,72,80,81,85,89,102,103,112,118,120,143,163]

Slices=np.load(WormConfig['working directory']+'/'+'Slices.npy')

mySlices=[Slices[n] for n in my_marker_no]

#%%
#img=Slices[18]
#img=Slices[16]
#img=img.astype(np.float)
#
#toatsu=threshold_otsu(img)
#timg=img.copy()
#timg[timg<toatsu]=0
#timg[timg>=toatsu]=1
#pos=np.where(timg==1)
#X=np.vstack(pos).transpose()
#X=np.vstack([X,X,X])
#plt.imshow(timg)
#
#idx,idy=np.indices(img.shape)
#pos=np.vstack([idx.flatten(),idy.flatten()]).transpose()

#%%
#Try with intensity
#img=Slices[68]
#img=img.astype(np.float)
#img/=img.mean()#eh
#idx,idy=np.indices(img.shape)
#pos=np.vstack([idx.flatten(),idy.flatten()]).transpose()
#X=np.transpose(np.vstack([mat.flatten() for mat in [idx,idy,img]]))
#timg=np.zeros(img.shape)

#%%

from skimage.draw import circle
def getInteriorCircleInfo(img,center,radius):
    x,y=circle(center[0],center[1],radius)#y first
    val=img[x,y]
    return x,y,val

search_radius=neuron_radius/2
search_idx=circle(neuron_center[0],neuron_center[1],search_radius)
#%%

#from scipy.stats import entropy

new_marker=[]

for img in mySlices:
    _,_,val=getInteriorCircleInfo(img,neuron_center,neuron_radius)

    #Hentropy=np.ones(img.shape)
    val=val.reshape(len(val),1)
    
    _,_,val=getInteriorCircleInfo(img,neuron_center,neuron_radius)
    
    fiftypercent=np.percentile(val,50)
    topval=val[val>fiftypercent][:,np.newaxis]
    
    #topval=val[:,np.newaxis]
    
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(topval)
    
    maxscore=-1*float('inf')
    new_center=[]
    for cx,cy in zip(search_idx[0],search_idx[1]):#row,column again
        _,_,newval=getInteriorCircleInfo(img,(cx,cy),neuron_radius)#accesses by row x column
        new_val=newval[:,np.newaxis]
        score=np.sum([math.exp(k) for k in kde.score_samples(new_val)])
        if score>maxscore:
            maxscore=score
            new_center=(cx,cy)            
    new_marker.append(new_center)
#    redraw=drawCircle(img,new_center,neuron_radius,img.max()+img.std(),1)
#    draw=drawCircle(img,neuron_center,neuron_radius,img.max()+img.std(),1)
#    redraw=drawCircle(draw,new_center,neuron_radius,img.max()+img.std(),1)
    #Hentropy[cx,cy]=np.sum(kde.score_samples(new_val))    
    
#new_val=newval[:,np.newaxis]
#cx,cy=(search_idx[0][0],search_idx[1][0])
#_,_,newval=getInteriorCircleInfo(img,(cx,cy),neuron_radius)
#new_val=newval[:,np.newaxis]

#%
#
draw_list=[drawCircle(s,neuron_center,neuron_radius,s.max()+s.std(),1) for s in mySlices]
#redraw_list=[drawCircle(s,best_center,neuron_radius,s.max()+s.std(),1) for s in draw_list]
#sidebysideplot(draw_list)
redraw_list=[drawCircle(s,c,neuron_radius,s.max()+s.std(),1) for s,c in zip(draw_list,new_marker)]

squarebysideplot(draw_list,map(str,my_marker_no))
squarebysideplot(redraw_list,map(str,my_marker_no))



#_,_,val=getInteriorCircleInfo(img,neuron_center,neuron_radius)
#
##Hentropy=np.ones(img.shape)
#val=val.reshape(len(val),1)
#
#_,_,val=getInteriorCircleInfo(img,neuron_center,neuron_radius)
#
#fiftypercent=np.percentile(val,50)
#topval=val[val>fiftypercent][:,np.newaxis]
#
#true_center=(25,35)#reverse from imagej
#_,_,valneuron=getInteriorCircleInfo(img,true_center,neuron_radius)
#
#binwidth=1000
#bins=range(np.min(img), np.max(img) + binwidth, binwidth)
#
#Data=(val,topval,valneuron)
#titles=['val','topval','valneuron']
#for i,d in enumerate(Data):
#    plt.figure()
#    plt.hist(d)
#    plt.title(titles[i])
#
#plt.figure()
#common_params=dict(normed=True)
#plt.hist(Data,bins=bins,normed=True)
#plt.legend(['initial circle','tophalf init circle','true neuron'])

#%%

#img2=Slices[163]
#cents=[(15,13),(35,30),(56,56)]
#valts=[getInteriorCircleInfo(img2,ce,neuron_radius)[2] for ce in cents]
#
#plt.hist(valts,normed=True,color=['r','g','b'],bins=10)


#%%

#def drawCircle(img,center,radius,color,thickness=1):#(row,col) format to reduce confusio
#    transpose_center=(center[-1],center[0])
#    return cv2.circle(img.copy(),center=transpose_center,radius=radius,color=color,thickness=thickness)



#redraw=drawCircle(img,(15,50),neuron_radius,img.max()+img.std(),1)    
#plt.figure()
#plt.imshow(redraw)
#
#
#
##topval=val[:,np.newaxis]
#
#kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(topval)
#
#maxscore=-1*float('inf')
#new_center=[]
#for cx,cy in zip(search_idx[0],search_idx[1]):
#    _,_,newval=getInteriorCircleInfo(img,(cx,cy),neuron_radius)
#    new_val=newval[:,np.newaxis]
#    score=np.sum([math.exp(k) for k in kde.score_samples(new_val)])
#    if score>maxscore:
#        maxscore=score
#        new_center=(cx,cy)            
#new_marker.append(new_center)
#redraw=drawCircle(img,new_center,neuron_radius,img.max()+img.std(),1)




#%%
def NTT(num_list):#(Numbers To Title)
    S=list(map(str,num_list))
    long_string=''
    while len(S)>0:
        long_string+=S.pop(0)
        long_string+=','
    return long_string[:-1]

def save_bad_marker(np_img_list,marker_no_list):
    file_name=NTT(marker_no_list)
    dst=np.hstack([np_img_list[i] for i in marker_no_list])
    imsave(WormConfig['working directory']+'/'+file_name+'.tif',dst,compress=1)
    
my_marker_no=[68,72,80,81,85,89,102,103,112,118,120,143,163]
S=list(map(str,my_marker_no))
#dst=np.hstack([draw_list[0],redraw_list[0]])
dst=np.hstack([Slices[i] for i in my_marker_no])
imsave(WormConfig['working directory']+'/'+NTT(my_marker_no)+'.tif',dst,compress=1)

###Cool parameter saving method
##common_params['normed'] = True
#plt.hist(val,**common_params)