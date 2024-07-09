# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 17:49:19 2015

@author: cgs567
"""

import os
os.listdir

from WormBox.ioworm import ls

from skimage.io import imread as sk_imread


direc='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/HeadRegion'

files=ls(direc)

bag=[sk_imread(f) for f in ls(direc) if f[-4:]=='.tif']


