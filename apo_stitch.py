# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 05:01:51 2015

@author: cgs567
"""



f=open('test_atlas.apo','w')

for i,nuc, pos in zip(ID,names,zip(*loc)):
    xx,yy,zz=pos
    v3d_line=str(i)+',  '+str(i)+', '+nuc+', , '+str(zz)+','+str(yy)+','+str(zz)+'0.000,0.000,0.000,50.000,0.000,,,,255,0,0\n'
    f.write(v3d_line)
f.close()