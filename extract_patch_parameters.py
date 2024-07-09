# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 01:53:34 2015

@author: csnyder
"""

from WormBox.ioworm import ls ,easy_contents,convert_index
import skimage.io as io


#='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/patches'
#path='/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/patches/patch'+str(marker)+'.tif'

readdir='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/seg_patches'
marker=22
path=readdir+'/'+'seg_Patch'+convert_index(marker,3)+'.tif'
I=io.imread(path)
from mayavi import mlab
mlab.pipeline.volume(mlab.pipeline.scalar_field(I))








