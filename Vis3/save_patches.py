# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:01:37 2015

@author: cgs567
"""

#from WormBox.ioworm import ls,return_those_ending_in
#from WormBox.WormFuns import getRegionBdry,contiguous_image_at_grid,returnGrid
#from WormBox.ioworm import getPIMWormData, sort_nicely
import numpy as np
import skimage.io as io
#try:
#    from skimage.filters import threshold_otsu,gaussian_filter
#except: #is python3
#    from skimage.filter import threshold_otsu,gaussian_filter
import skimage.measure as m #for connected components





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
def parse_line(line,file_extension):
    if file_extension=='apo':
        return parse_apo_line(line)
    elif file_extension=='marker':
        return np.array(parse_marker_line(line))
    else:
        raise Exception('file_extension not .apo or .marker')
        

def get_number(line):#only works with apo right now sorry
    return line.split(',')[0]

        
def getLocalImage(marker_loc,v,window_radius):
#    if out_of_bounds(marker_loc):
#        raise Exception('Marker loc + window radii goes beyond edge of image')
    low,high=marker_loc-window_radius,marker_loc+window_radius+1
    if isinstance(v,np.ndarray):
        return v[low[0]:high[0],low[1]:high[1],low[2]:high[2]]
    else:#is pims
        return np.array( [ _v[low[0]:high[0],low[1]:high[1]] for _v in v[low[-1]:high[-1]] ] )        


    

if __name__=='__main__':
    #####Careful about overwrite!!!#####
    #windows_file_name='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150302_CM_Hi-res_40x/W5/150226_ZIM294_L4_W5 neuron_location_chris.apo'
    #stack_path='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150302_CM_Hi-res_40x/W5/150226 Hi-res_40x Full Worm W5.tif'
    #dst_folder='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150302_CM_Hi-res_40x/W5/patches'
    
    
#    windows_file_name='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/truncatedStitch/150303_CM_Hi-res_20x/2/150303_CM_Hi-res_20x annotation chris on 150618.marker'
#    stack_path='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/truncatedStitch/150303_CM_Hi-res_20x/2/150303_ZIM294_L4_W2 Full Worm.tif'
#    dst_folder='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/truncatedStitch/150303_CM_Hi-res_20x/2/patches'
#    
    
    read_dir='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/'
#    write_dir='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/full_segmentation'
    marker_file='150303_CM_Hi-res_20x annotation chris on 150618.apo'
    
    file_name=read_dir+marker_file
    
#    file_name=windows_file_name
    f=open(file_name,'r')
    
    file_extension=file_name.split('.')[-1]    
    
    
    marker_list=[[get_number(line),parse_line(line,file_extension)] for line in f if line[0]!='#']#in x, y, z
    
    
    
    #stack_path='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal/150226_CM_Hi-res_40x/W5/stacks/150226 Hi-res_40x W5.tif'
    ##'./../Corral/Snyder/WormPics/150226_CM_Hi-res_40x/W5/stacks\\150226 Hi-res_40x W5.tif'
    #
    #v=skio.imread(stack_path)#only works in python2 #otherwise need MultiImage class
    ##v=v.transpose((0,2,1))
    #
    #
#    z_range=[min(list(zip(*marker_list))[2]),max(list(zip(*marker_list))[2])]
    neuron_window=np.array([75,75,35])#must be odd
    window_radius=(neuron_window-1)//2#x,y,z
    
    
    
#    X=io.imread(stack_path)
    
    
#    for i,mark in enumerate(marker_list):
#    #mark=marker_list[0]
#    
#        temp_mark=np.array([mark[2],mark[1],mark[0]])#Just depends on the format of image X
#        temp_window_radius=np.array([window_radius[2],window_radius[1],window_radius[0]])
#        patch=getLocalImage(temp_mark,X,temp_window_radius)
        
        
#        io.imsave(dst_folder+'/patch'+str(i)+'.tif',patch)
    
    
        
    
    f.close()
    
    
    
    
    
    #########
    #rel_marker_coord=(neuron_window-1)/2#coordinate of marker inside neuron_window volume
    #r_row,r_col,r_z=window_radii
    
    #neuron_radius=12
    #neuron_center=(34,34)