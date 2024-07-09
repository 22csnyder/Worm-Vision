# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 20:06:47 2015

@author: melocaladmin
"""


import matplotlib.pyplot as plt
import numpy as np
import cv2
#import pims
#import matplotlib.gridspec as gridspec
#from WormFuns import stackplot, adjust
#from WormFuns import *
#from formatworm import *
#from WormPlot import *

from ioworm import *


foldername="C:\\Users\\melocaladmin\\Documents\\WormPics\\141226_CM_Pan_GCamp_Vol1_Stacks\\"
time_stack_file="c_141226_ZIM294 Vol1_L4_O20X_F630x75um_Vol_P6144x500S28_g2_stack1of7.tiff"
subfolders=['stack1split','stack2split','stack3split','stack4split','stack5split','stack6split','stack7split']
#v=time_stack_read()#Doesn't work! Stack too big

all_files=[]    
for sub in subfolders:
    filenames=ls(foldername+sub)
    all_files.extend(filenames)
sort_nicely(all_files)#alphanumeric sorting

sz=all_files.__len__()

t_dim=50*7
z_dim=10
indicies=np.resize(np.linspace(0,sz-1,sz).astype('int'),(t_dim,z_dim))

total_light=[]

#f=open('total_volume_integral','w')
#for idx in indicies:
#    lgt=0    
#    for i in idx:
##        print i
#        X=cv2.imread(all_files[i],0)
#        lgt+=np.sum(X,dtype=np.uint16)        
#    total_light.append(lgt)
#    f.write('%d %d \n' % (idx[0]/10,lgt))
#
#f.close()


data=np.loadtxt(foldername + 'total_volume_integral')
(time,Light)=zip(*data)       
#Light=np.array(total_light).astype(np.float32)
percent_change_Light=100*(Light-np.mean(Light))/np.mean(Light)

time=np.linspace(0,120,t_dim)
plt.plot(time,percent_change_Light)
plt.xlabel('Time (s)')
plt.ylabel('Percent Intensity Change (relative to Average)')
plt.title('Total Volume Intensity Change over Time')
plt.show(block=False)

