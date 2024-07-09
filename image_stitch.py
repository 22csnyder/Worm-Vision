# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 00:08:15 2015

@author: cgs567
"""

#import matplotlib.pyplot as plt
#from WormBox.ioworm import ls
#from WormBox.WormPlot import sidebysideplot
import numpy as np
from skimage import io
#from tifffile import imsave
from WormBox.BaseClasses import Environment, Worm


#stitchfile='/StitchConfiguration.txt'
stitchfile='/StitchConfiguration2.txt'

#main_dir='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/20x/W2/'

#main_dir='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/'
main_dir='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W2/'

env=Environment(main_dir)
#f=open(WormConfig['working_directory']+stitchfile,'r')

f=open(env.read_dir+stitchfile,'r')

class metaData():
    def __init__(self, name,coordinates):
        self.name=env.read_dir+name
#        self.name=WormConfig['working_directory']+'/'+name
        self.x,self.y,self.z=coordinates
        self.vec=np.array(coordinates)


def parse_line(line):
    if line[0]=='#':
        return
    else:
        start=line.find('(')
        end=line.find(')')
        numbers=line[start+1:end]
        coord=numbers.split(', ')
        file_name=line.split('; ;')[0]
    return metaData(file_name,[int(c) for c in coord])

meta=[parse_line(l) for l in f if parse_line(l) is not None]


print "reading ",len(meta)," images..."
v=[io.imread(m.name) for m in meta]
print "complete"


print 'concatenating..'
offset=meta[1].vec-meta[0].vec

vx,vy,vz=offset
wz,wy,wx=np.array(v[1].shape)-np.array(v[0].shape)
print "offset (right-left) is ", offset
#offset=offset[1:]#ignore first coordinate which is x

pad_left= ( (max(   vz,0),max(0,wz-vz) ), (  max(   vy,0), max(0,wy-vy) ) , (0,0))
pad_right=( (max(-1*vz,0),max(0,vz-wz) ), (  max(-1*vy,0), max(0,vy-wy) ) , (0,0))


left=np.lib.pad(v[0],pad_left,'constant')#zyx in relation to xyz format from imageJ
right=np.lib.pad(v[1],pad_right,'constant')
stitched_image=np.dstack([left,right])

##print 'complete'

##print 'saving...'

#io.imsave(WormConfig['working_directory']+'/'+'multipage.tif', stitched_image)

worm=Worm(env,stitched_image)
#worm.Stack=stitched_image
worm.save('multipage.tif')

f.close()







##############################

##'deflate compression not supported' --vaa3d --have to load into vaa3d as slices then save as RAW
#imsave(WormConfig['working_directory']+'/'+'multipage.tif', stitched_image,compress=1)
#
##imsave(WormConfig['working_directory']+'/'+'test.tif', stitched_image,compress=1)#no compression
##http://en.wikipedia.org/wiki/Tagged_Image_File_Format
#
#



#    else: return list(map(int,line.split(',',-1)[0:3]))
#marker_list=[parse_marker_line(line) for line in f][1:]#in row, col, z
#
#z_range=[min(list(zip(*marker_list))[2]),max(list(zip(*marker_list))[2])]
#
#window_radii=np.array([15,15,2])
#r_row,r_col,r_z=window_radii