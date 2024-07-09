# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 12:03:37 2015

@author: csnyder
"""

import cPickle as pickle
import dill
from WormBox.ioworm import ls,return_those_ending_in,convert_index,sort_nicely,get_file_index
from WormBox.BaseClasses import Environment
from skimage import io
import numpy as np
from save_patches import parse_line,get_number
#from volume_views import Vis




def get_segment_list(path):
    files=ls(path)
    files.sort(key=get_file_index)
    segments=[]
    for f in files:
        with open(f,'r') as handle:
            segments.append(dill.load(handle))
    return segments


if __name__=='__main__':
    
    ####
    
    
    #data='150303'
    #data='150731'
    #  
    #data='150311_20x_W1-1'
    
    data='150311_40x_W1'
    
    
    #Data from patches
    
    if data=='150311_20x_W1':
        seg_path1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/20x/W1/Segments/W1-1'
        seg_path2='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/20x/W1/Segments/W1-2'
        savefile='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/20x/W1/150311_ZIM294_L4_W1_O20X_152608_seg.tif'
        segments1=get_segment_list(seg_path1)
        segments2=get_segment_list(seg_path2)
        seg1=segments1[0]
        seg2=segments2[0]
        segments=segments1+segments2
    
    elif data=='150311_40x_W1':
        seg_path1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/Segments/W1-1'
        seg_path2='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/Segments/W1-2'
        seg_path3='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/Segments/W1-3'
        
        savefile='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/150311_ZIM294_L4_W1_40X_152608_seg.tif'
        segments1=get_segment_list(seg_path1)
        segments2=get_segment_list(seg_path2)
        segments3=get_segment_list(seg_path3)
        seg1=segments1[0]
        seg2=segments2[0]
        seg3=segments3[0]
        segments=segments1+segments2+segments3    
    
    
    centers=[s.offset for s in segments]
    z,y,x=zip(*centers)
    zlim,ylim,xlim=np.array([min(z),max(z)]),np.array([min(y),max(y)]),np.array([min(x),max(x)])
    
    border=[100,100,100]
    global_offset=np.array([ border[0]-zlim[0]  ,   border[1] - ylim[0] ,   border[2] - xlim[0] ])
    
    shape=global_offset+[zlim[1],ylim[1],xlim[1]]+border#vis.I.shape is z,y,x format
    
    #neuron_window=np.array([75,75,35])
    #window_radius=np.array((neuron_window-1)//2)#x,y,z   
    window_radius=np.array([17,37,37])#z,y,x#equivalent to above definition
    
    def get_patch_intensity(seg):
        p=seg.binary_patch.astype(np.int)
        q=seg.float_patch
        return np.mean(q[p==1])
    
    print 'forming array'
    
    Y=np.zeros(shape)
    
    for s in segments:
        low=global_offset+s.offset-window_radius
        high=global_offset+s.offset+window_radius#+1
        Y[low[0]:high[0],low[1]:high[1],low[2]:high[2]]+=s.binary_patch.astype(np.int)
    
    Y[Y>1]=1
    
    #from mayavi import mlab
    #mlab.pipeline.volume(mlab.pipeline.scalar_field(Y))
    #mlab.title(data,size=0.35)
    
    
    
    io.imsave(savefile,Y)
    
    
    #p=seg0.binary_patch.astype(np.int)
    #mlab.pipeline.volume(mlab.pipeline.scalar_field(q))
    #q=seg0.float_patch
    
    #intensities=[get_patch_intensity(s) for s in segments]
    #
    #
    #
    #for s in segments:
    #    print s.offset
    #    print s.notes
    
    
    #read_dir='/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/patches/'
    #env=Environment(read_dir)
    # 
    #
    #p=io.imread(env.read_dir+'patch47.tif')
    #
    #path='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/Segmentations/150721_chris_seg_patches/'
    #q=io.imread(path+'seg_Patch086.tif')
    #
    #apo='150303_CM_Hi-res_20x annotation chris on 150618.apo'
    #
    #apo_file=path+apo
    #
    #with open(path+apo,'r') as f:
    #    file_extension=apo_file.split('.')[-1]
    #    raw_list=[[get_number(line),parse_line(line,file_extension)] for line in f if line[0]!='#']#in x, y, z
    ##    for label,marker in raw_list:
    ##        self.segments[label]=dict(marker=marker)
    #
    #names,markers=zip(*raw_list)
    #
    #window_radius=np.array([37,37,17])####Hope this never changes because it's everywhere
    #offsets=[ (m-window_radius)[::-1] for m in markers]
    #
    #whole_worm_path='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/150303_ZIM294_L4_W2 Full Worm.tif'
    #I0=io.imread(whole_worm_path)
    #
    #for n,offset in zip(names[:3],offsets[:3]):
    #    name=convert_index(n,3)
    #    f_name=path+'seg_Patch'+name+'.tif'
    #    P=io.imread(f_name)
    #    loc0,loc1,loc2=np.where(P==1)
    #    loc0+=offset[0];loc1+=offset[1];loc2+=offset[2]
        
        
    
    
    
    
    
    #filename='first_orddict_pickle.p'
    #   
    #read_dir='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/Segmentations/1/'
    ##write_dir='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/Segmentations/1/'
    #env=Environment(read_dir=read_dir)
    #
    ##Data from dill pickling
    #filename=env.read_dir+'first_orddict_pickle.p'
    #dill_files=return_those_ending_in(ls(env.read_dir),'pkl')
    #data=[]
    #for f in dill_files:
    #    with open(f,'r') as handle:
    #        data.append(dill.load(handle))
    
    
