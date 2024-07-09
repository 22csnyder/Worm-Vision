# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 08:34:22 2015

@author: csnyder
"""
import pims
import numpy as np
import os
import skimage.io as io



class Environment(object):
    Initialized_Environment=False
    def __init__(self,read_dir,write_dir=None):
        self.read_dir=read_dir
        self.write_dir=write_dir
        if self.write_dir==None:
            self.write_dir=self.read_dir
        self.Initialized_Environment=True   

class Worm4D(Environment):
    def __init__(self,hyperstack_path,slices_per_stack,*args):
        self.sps=slices_per_stack
        #init environment
        if args and isinstance(args[0],Environment):
            Environment.__init__(self,args[0].read_dir,args[0].write_dir)
            self.Initialized_Environment=True
        else:
            #Try to guess what environment is
            try:
                loc=hyperstack_path.rfind('/') #Broken if ends with '/'
                read_dir=hyperstack_path[:loc]
                Environment.__init__(self,read_dir=read_dir,write_dir=read_dir)
                self.Initialized_Environment=True
            except ValueError:
                print 'relative path given without environment being given'


        if '/' not in hyperstack_path:#relative path
            try:
                hyperstack_path=self.read_dir+'/'+hyperstack_path
            except ValueError:
                print 'relative path given but environment not initialized yet'




            
        self.hyperstack_path=hyperstack_path
        self.slices_per_stack=slices_per_stack

        if os.path.isfile(hyperstack_path):
            self.HyperStack=pims.TiffStack(hyperstack_path,process_func=lambda frame:np.rot90(frame,3))
        else:
            print 'class only fit to handle single stack, not directory of worms'

        self.n_slices=len(self.HyperStack)
        self.n_worms=self.n_slices//self.sps

    def load_data(self):
        if isinstance(self.HyperStack,np.ndarray):
            pass
        else:
            self.HyperStack=io.imread(self.hyperstack_path)

    
    def spawn_worm(self,idx):
        if idx>=self.n_worms:
            raise ValueError("worm idx is greater than hyper.n_worms")
        worm=Worm(Environment(self.read_dir,self.write_dir))
        a=self.sps*idx;  b=self.sps*(idx+1)
        if isinstance(self.HyperStack,np.ndarray):
            worm.Stack=self.HyperStack[a:b,...]
        else:#pims
            V=[self.HyperStack[i] for i in range(a,b)]
            worm.Stack=np.vstack([v[np.newaxis,:] for v in V])
        return worm


class Worm(Environment):
    Stack=None
    def __init__(self,*args):
        if isinstance(args[0],Environment):
            Environment.__init__(self,args[0].read_dir,args[0].write_dir) 
        if isinstance(args[1],np.ndarray):
            self.Stack=args[1]
            
    def set_stack_path(self,stack_path):
        self.stack_path=stack_path        
        loc=stack_path.rfind('/')#finds last /
        
        self.data_set=stack_path[loc+1:]
        if not self.Initialized_Environment:
            self.read_dir=stack_path[:loc]
            self.write_dir=self.read_dir
            self.Initialized_Environment=True
        self.Stack=pims.TiffStack(self.stack_path,process_func=lambda frame:np.rot90(frame,3))    

    def load_data(self,file_name):
        self.Stack=io.imread(self.read_dir+file_name)
    def save(self,name):
#        if isinstance(self.Stack,np.ndarray):
        if name[-4:] != '.tif':
            name+='.tif'
        io.imsave(self.write_dir+name,self.Stack) #assume name ends in .tif


class Nucleus(Environment):
    Patch=None
    def __init__(self,*args):
        if isinstance(args[0],Environment):
            Environment.__init__(self,args[0].read_dir,args[0].write_dir)        

    def load_data(self,file_name):
        self.Patch=io.imread(self.read_dir+file_name)








if __name__=='__main__':
    env=tEnvironment('a','b')





#    def save(self,name):
#<<<<<<< HEAD
#        if len(name)<4:
#            name+='.tif'
#        if len(name)<5:
#            name+='.tiff'
#        if name[-4:]!='.tif' and name[-5:]!='.tiff':
#            name+='.tif'
#        io.imsave(self.write_dir+'/'+name,self.Patch)
#=======
#        io.imsave(self.write_dir+name+'.tif',self.Patch)
#>>>>>>> 401f8d2f3533568d438764d14beb8d94d799dcf8


#class nucleus(Worm):
#    def __init__(self,number):
#        self.Slice=self.Slices[number]
#        self.Patch=self.Patches[number]




        
#        self.Patches=delayed(skio.imread,self.data_set)

        #Do lazy loading of stack
        #assumes for now that stack is 3d        

      

        
        
#    markers
#    Slices
#    Patches

#    'WormPics directory':pathWormPics,
#    'working directory':pathWormPics+'/Synthesized/truncatedStitch/150226 Hi-res_40x/W5',
#    'data set identifier':
