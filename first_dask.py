# -*- coding: utf-8 -*-
"""
Created on Sun Sep 06 01:07:23 2015

@author: cgs567
"""

#from dask.array.image import imread
import numpy as np
import dask as da
#from dask.core import Array
from dask.array import Array
from skimage.io import imread as sk_imread
from glob import glob
import os



#tiff_dir='C:/Users/cgs567/Documents/Corral/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/TIFF/'
#small_tiff_dir='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/first three stacks/'
#re_name = small_tiff_dir+'sc_150902_1703_Vol_Dev_II_No_Stimulus_W1_W1_ZIM294_L4_No_Stimulus_O20X_F630x150_P11008x500S188_*****.tiff'
#test_file=small_tiff_dir+'sc_150902_1703_Vol_Dev_II_No_Stimulus_W1_W1_ZIM294_L4_No_Stimulus_O20X_F630x150_P11008x500S188_00007.tiff'
#small_tiff_dir='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/first three stacks/'
#filename = small_tiff_dir+'sc_150902_1703_Vol_Dev_II_No_Stimulus_W1_W1_ZIM294_L4_No_Stimulus_O20X_F630x150_P11008x500S188_*****.tiff'

#x=sk_imread(test_file)

#im=imread(re_name)#chunks=(3,15,500,2100)
#A=im[:,5,5]



def add_leading_dimension2(x):
    return x[None,None, ...]
def add_leading_dimension(x):
    return x[None, ...]
def tokenize(*args):
    from hashlib import md5
    return md5(str(args).encode()).hexdigest()


f0='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/first 4 stacks/sc_150902_1703_Vol_Dev_II_No_Stimulus_W1_W1_ZIM294_L4_No_Stimulus_O20X_F630x150_P11008x500S188_00021.tiff'
x0=sk_imread(f0)

filename='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/first 4 stacks/sc_150902_1703_Vol_Dev_II_No_Stimulus_W1_W1_ZIM294_L4_No_Stimulus_O20X_F630x150_P11008x500S188_*****.tiff'


filenames = sorted(glob(filename))
if not filenames:
    raise ValueError("No files found under name %s" % filename)

name = 'imread-%s' % tokenize(filenames, map(os.path.getmtime, filenames))

sample = sk_imread(filenames[0])
#dsk = dict(((name, i) + (0,) * len(sample.shape),
#            (add_leading_dimension, (sk_imread, filename)))
#            for i, filename in enumerate(filenames))
#chunks = ((1,) * len(filenames),) + tuple((d,) for d in sample.shape)


Z=15
T=len(filenames)//Z
#(time,z,(imageshape))
#Key is a tuple (name,idx,idx,..)
dsk = dict(((name, i//Z, i%Z) + (0,) * len(sample.shape),
            (add_leading_dimension2, (sk_imread, filename)))
            for i, filename in enumerate(filenames))

#chunks = ((1,) * len(filenames),) + tuple((d,) for d in sample.shape)

chunks = ((1,) * T,) + ((1,) * Z,) + tuple((d,) for d in sample.shape)

D=Array(dsk, name, chunks, sample.dtype)
X=np.array(D)

from skimage import io
io.imsave('dasktest.tif',X)
#####try da.array( ) instead
#return Array(dsk, name, chunks, sample.dtype)

#im.chunks=()
#im.numblocks
#im.cache
#im.nbytes
#X=np.array(im)###Only do if data is small