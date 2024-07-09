# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 17:58:17 2015

@author: cgs567
"""
import skimage.io as io
import numpy as np

from WormBox.BaseClasses import Worm, Worm4D , Environment
from WormBox.ioworm import ls

#####EDIT HERE#####
input_directory='C:/Users/cgs567/Documents/Corral/Ki-Confocal-2015/150624_1653_Vol_Img_Prelim_Glycerol_1M/TIFF'
output_directory='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150624_1653_Vol_Img_Prelim_Glycerol_1M/Stacks'
stack_size=15
start_tag=34#Tags start with 1 instead of 0
#####################




increment_folders=ls(input_directory,verbose=0)

image_files=[]
for folder in increment_folders:#Make sure alphabetical order creates proper ordering
    image_files+=ls(folder,fullpath=1,verbose=0)




N=len(image_files)
end_tag= N - (N-start_tag)%stack_size
n_stacks=(end_tag-start_tag)/stack_size

select_image_files= image_files[start_tag-1 : end_tag-1]
sorted_image_files=[select_image_files[a*stack_size:(a+1)*stack_size] for a in range(n_stacks)]

print 'Begin saving stack'
for i,stack_file in enumerate(sorted_image_files):
    print i
    im_list=[io.imread(f) for f in stack_file]
    Stack=np.vstack([im[np.newaxis,:] for im in im_list])
    if i<10:
        pad='00'
    elif i<100 and i>=10:
        pad='0'
    elif i>=100:
        pad=''
    io.imsave(output_directory+'/'+'Stack'+pad+str(i)+'.tif',Stack)



#inspect to make sure in correct order
#tags= [int(fi.replace('.','_').split('_')[-2]) for fi in image_files]





#
##Hope
#img_list=[io.imread(f) for f in image_files]
#print('Images loaded')
#Stack=np.dstack(img_list)
#print('Stack Concatenated')
#io.imsave(output_directory+'/'+'AllSlicesConcatenated.tif',Stack)
#print('im saved!')

