# -*- coding: utf-8 -*-
"""
Created on Sun Sep 06 03:21:42 2015

@author: cgs567
"""

import numpy as np
import os
import skimage.io as io


#from dask.array.image import imread
#import numpy as np
#import dask as da
#from dask.core import Array
from dask.array import Array
from skimage.io import imread as sk_imread
from glob import glob
import os
from mayavi.sources.api import ArraySource
from mayavi.modules.volume import Volume


def add_leading_dimension2(x):
    return x[None,None, ...]
def add_leading_dimension(x):
    return x[None, ...]
def tokenize(*args):
    from hashlib import md5
    return md5(str(args).encode()).hexdigest()

##Reproduced in below function: consider delete
#filename='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/first 4 stacks/sc_150902_1703_Vol_Dev_II_No_Stimulus_W1_W1_ZIM294_L4_No_Stimulus_O20X_F630x150_P11008x500S188_*****.tiff'
#filenames = sorted(glob(filename))
#if not filenames:
#    raise ValueError("No files found under name %s" % filename)
#name = 'imread-%s' % tokenize(filenames, map(os.path.getmtime, filenames))
#sample = sk_imread(filenames[0])


#dsk = dict(((name, i) + (0,) * len(sample.shape),
#            (add_leading_dimension, (sk_imread, filename)))
#            for i, filename in enumerate(filenames))
#chunks = ((1,) * len(filenames),) + tuple((d,) for d in sample.shape)


def get_dask_array(regex_filename,stack_size):##Feed in ex: get_dask_array(myimage***.tif,15)
    filenames = sorted(glob(regex_filename))
    if not filenames:
        raise ValueError("No files found under name %s" % regex_filename)
    name = 'imread-%s' % tokenize(filenames, map(os.path.getmtime, filenames))
    sample = sk_imread(filenames[0])#read one file just so we can get an idea of the dims
    Z=stack_size
    T=len(filenames)//Z
    #Key is a tuple (name,idx,idx,..)
    dsk = dict(((name, i//Z, i%Z) + (0,) * len(sample.shape),
                (add_leading_dimension2, (sk_imread, filename)))
                for i, filename in enumerate(filenames))
    chunks = ((1,) * T,) + ((1,) * Z,) + tuple((d,) for d in sample.shape)#Stored in memory file-wise. Make more flexible in future
    return Array(dsk, name, chunks, sample.dtype)



class Environment(object):
    Initialized_Environment=False
    def __init__(self,read_dir,write_dir=None):
        self.read_dir=read_dir
        self.write_dir=write_dir
        if self.write_dir==None:
            self.write_dir=self.read_dir
        self.Initialized_Environment=True


def system_dependent_directories():
    if os.name is 'nt':
        ki_dir='C:/Users/cgs567/Documents/Corral/Ki-Confocal-2015/'
        snyder_dir='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/'
    elif os.name is 'posix':
        ki_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/'
        snyder_dir='/work/03176/csnyder/Corral/Snyder/'
    return ki_dir,snyder_dir

class DirectoryResource(object):
    read_dir=None
    write_dir=None
    def __init__(self,dataset):
        self.ki_dir,self.snyder_dir=system_dependent_directories()
        self.read_dir=self.ki_dir+dataset
        self.write_dir=self.snyder_dir+dataset
#        self.get_ini_file()
#    
#    def is_ini(self,f):
#        try:
#            return f[-4:]=='.ini'
#        except:
#            return False
#    def get_ini_file(self):
#        self.lvl1=os.listdir(self.read_dir)
#        self.ini_file=[f for f in self.lvl1 if self.is_ini(f)][0]

class Worm(Environment):
    Stack=None
    regex_file=None
    spacing=None
    def __init__(self,**kwarg):
        pass
    def dask_read_data(self,re_file,stack_size):
        if hasattr(self,'read_dir'):
            re_file=self.read_dir+re_file
        self.daX=get_dask_array(re_file,stack_size)##t,z,y,x 


#        if isinstance(args[0],Environment):
#            Environment.__init__(self,args[0].read_dir,args[0].write_dir) 
#        if isinstance(args[1],np.ndarray):
#            self.Stack=args[1]
            
#    def set_stack_path(self,stack_path):
#        self.stack_path=stack_path        
#        loc=stack_path.rfind('/')#finds last /
#        
#        self.data_set=stack_path[loc+1:]
#        if not self.Initialized_Environment:
#            self.read_dir=stack_path[:loc]
#            self.write_dir=self.read_dir
#            self.Initialized_Environment=True
#        self.Stack=pims.TiffStack(self.stack_path,process_func=lambda frame:np.rot90(frame,3))    

    def plane_by_plane(self,z):
        return np.array(self.daX[:,z,:,:])
    def load_data(self,file_name):
        self.Stack=io.imread(self.read_dir+file_name)
    def save(self,name):
#        if isinstance(self.Stack,np.ndarray):
        if name[-4:] != '.tif':
            name+='.tif'
        io.imsave(self.write_dir+name,self.Stack) #assume name ends in .tif

    def display_on_scene(self,scene,t=0):#t is for default time point
        self.array_src=ArraySource(spacing=self.spacing)
        try:
            self.array_src.scalar_data=self.daX[t].compute()
        except:
            self.array_src.scalar_data=self.daX[t]#is already a numpy array
            
        scene.mayavi_scene.add_child(self.array_src)
        vol_module=Volume()
        self.volume=self.array_src.add_module(vol_module)
#        self.add_trait('volume',self.array_src.add_module(vol_module))

if __name__=='__main__':     
    #'C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/first 4 stacks/sc_150902_1703_Vol_Dev_II_No_Stimulus_W1_W1_ZIM294_L4_No_Stimulus_O20X_F630x150_P11008x500S188_00021.tiff'
    #rd='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/first 4 stacks/'
    
    #rd='C:/Users/cgs567/Documents/Corral/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/TIFF/'
    wd='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/'
    
    dataset='150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/TIFF/'
    re_file='sc_150902_1703_Vol_Dev_II_No_Stimulus_W1_W1_ZIM294_L4_No_Stimulus_O20X_F630x150_P11008x500S188_*****.tiff'
    
    io_pal=DirectoryResource(dataset)
    
    worm=Worm()
    worm.read_dir=io_pal.read_dir
    worm.dask_read_data(re_file,15)
    
    plane=9
    P=worm.plane_by_plane(plane)
    io.imsave(wd+'plane'+str(plane)+'.tif',P)



