# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 18:37:57 2014

@author: melocaladmin
"""

#import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#openworm data straightened by Christian Grove
#Grove='./../WormData\\Straightened_Neuron_Locations.txt'
Grove='C:/Users/cgs567/Documents/WormData/Straightened_Neuron_Locations.txt'
#dataloc="C:\Users\melocaladmin\My Documents\WormData"
#filename="\Straightened_Neuron_Locations.txt"


def getGroundTruth(f_name):
    fo=open(f_name,"r")
    if fo.closed:
        print("bad filename")
        return
        
    data=fo.readlines()
    
    x,y,z=[],[],[]
    nn=[]
    #print data
    
    for line in data:
        line=line.strip()
        words=line.split()
        name=words[0]
        xpos=float(words[2].strip("(,"))
        ypos=float(words[3].strip(","))
        zpos=float(words[4].strip(")"))
        nn.append(name)
        x.append(xpos)
        y.append(ypos)
        z.append(zpos)
    fo.close()
    
    
    return nn,[x,y,z]
    
######Main######
import numpy as np
_names, _loc=getGroundTruth(Grove)
ID=list(range(1,_names.__len__()+1))

nuc='first_nuc'

###This part converts the data file to a .apo file
#import os
#os.chdir(Grove+'/../')
#f=open('test_atlas.apo','w')
#for i,nuc, pos in zip(ID,_names,zip(*_loc)):
#    xx,yy,zz=pos
#    v3d_line=str(i)+',  '+str(i)+', '+nuc+', , '+str(zz)+','+str(yy)+','+str(zz)+'0.000,0.000,0.000,50.000,0.000,,,,255,0,0\n'
#    f.write(v3d_line)
#f.close()


def is_sorted(L):
    for i in range(len(L)-1):
        if L[i]>L[i+1]:
            return False
    return True



_x=_loc[0];_y=_loc[1];_z=_loc[2]


ans=sorted(zip(_names,_x,_y,_z),key=lambda tup:tup[2])
names,x,y,z=zip(*ans)

xh,yh,zh=x[:120],y[:120],z[:120]
xt,yt,zt=x[120:],y[120:],z[120:]

#
#print location
fig = plt.figure()
ax=fig.add_subplot(111,projection='3d',aspect='auto')

#print x
ax.scatter(xh,yh,zh,c='b',marker='o')
ax.scatter(xt,yt,zt,c='r',marker='o')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("Ground Truth ('Head' in blue)")
ax.set_ylim([-4.5,-3])

plt.show()








