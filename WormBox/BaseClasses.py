# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:26:58 2015

@author: cgs567
"""
import pims
import numpy as np
import os
import skimage.io as io

from traits.api import HasTraits,Str,Array,Instance

from collections import OrderedDict 


def dir_format(directory):  #Maybe apply this everywhere
    if directory[-1] is not '/':
        directory+='/'
    return directory



class ApoReader(HasTraits):
    filename=None
    marker_list=None
    def __init__(self,*args):#filename=None,marker_list=None):
        if isinstance(args[0],str):
            self.filename=args[0]
        elif isinstance(args[0],list):
            self.marker_list=args[0]

    @staticmethod
    def get_number(line):#only works with apo right now sorry
        return line.split(',')[0]
    @staticmethod
    def parse_apo_line(line):
        pieces=line.split(',')
        z=int(pieces[4].strip())
        x=int(pieces[5].strip())
        y=int(pieces[6].strip())
        return [x,y,z]
    @staticmethod
    def parse_marker_line(line):
        if line[0]=='#':pass
        else: return list(map(int,line.split(',',-1)[0:3]))
    def parse_line(self,line):
        if self.file_extension in '.apo':
            return self.parse_apo_line(line)
        elif self.file_extension in '.marker':
            return np.array(self.parse_marker_line(line))
        else:
            raise Exception('file_extension not .apo or .marker')
    def read_markers(self):#Might still be in x,y,z format--backwards compatability
        with open(self.filename,'r') as f:
            self.file_extension=self.filename.split('.')[-1]
            self.marker_list=[self.parse_line(line) for line in f if line[0]!='#']#in x, y, z#row,col,z
            self.x,self.y,self.z=zip(*self.marker_list)
            self.x=np.array(self.x);self.y=np.array(self.y);self.z=np.array(self.z)

    def add_marker_file_to_dict(self,dictionary):
        with open(self.filename,'r') as f:
            self.file_extension=self.filename.split('.')[-1]
            raw_list=[[self.get_number(line),self.parse_line(line)] for line in f if line[0]!='#']#in x, y, z
            for label,marker in raw_list:
                dictionary['apo'+label]=dict(marker=marker[::-1])##Use z,y,x format here
    def add_marker_list_to_dict(self,dictionary):
        labels=[str(i) for i in range(len(self.marker_list))]
        for label,marker in zip(labels,self.marker_list):
            dictionary[label]=dict(marker=np.array(marker[::-1]))
    
    def add_data_to_dict(self,dictionary):
        if self.marker_list is not None:
            self.add_marker_list_to_dict(dictionary)
        else:
            if self.filename is not None:
                self.add_marker_file_to_dict(dictionary)
            else:
                raise ValueError('either marker_list or filename must be defined for ApoReader to add data')
                
DEFAULT_KEY=''
class Segments(OrderedDict):
    
#    def __init__(self):#######DEBUG commented out because break pickle
#        OrderedDict.__init__(self)
#        self.apo_reader=None
#    def read_markers(self,data):
#        self.apo_reader=ApoReader(data)
#        self.apo_reader.add_data_to_dict(self.segments)
        
        
#    def __getitem__(self,key):
#        if isinstance(key,int):
#            return self.values()[key]
#        else:
#            return self.dict[key]
    def new_neuron_at(self,point,label=None):
        if label is None:
            label=str(len(self.items()))
        self[label]=dict(marker=point)
        self.current_name=label
    def previous_key(self, key=DEFAULT_KEY):
        if key is DEFAULT_KEY:
            return self.first_key()
        previous = self._OrderedDict__map[key][0]
        if previous is self._OrderedDict__root:
            raise ValueError("{!r} is the first key".format(key))
        previous_key=previous[2]
        self.current_name=key        
        return previous_key

    def next_key(self, key=DEFAULT_KEY):
        if key is DEFAULT_KEY:
            return self.first_key()
        next_ = self._OrderedDict__map[key][1]
        if next_ is self._OrderedDict__root:
            raise ValueError("{!r} is the last key".format(key))
        next_key=next_[2]
        self.current_name=key
        return next_key
        
    def first_key(self):
        try:
            first_key=[key for key in self][0]
        except:
            raise ValueError("OrderedDict() is empty")
            return None
        self.current_name=first_key            
        return first_key

    def last_key(self):
        try:
            last_key=[key for key in self][-1]
        except:
            raise ValueError("OrderedDict() is empty")
            return None
        self.current_name=last_key            
        return last_key

    def get_next_item(self,starting_from=None):
        if self.key_is_last(starting_from):
            print 'Already at last item!'
            return False
        if starting_from is None:
            return self.items[0]#start from beginning
        else:
            idx=self.get_idx(starting_from)
            return self.items()[idx]

    def get_previous_item(self,starting_from=None):
        if self.key_is_first(starting_from):
            print 'Already at first item!'
            return False
        if starting_from is None:
            return self.items[0]#start from beginning
        else:
            idx=self.get_idx(starting_from)
            return self.items()[idx]
            
    
    def key_is_last(self,key):
        if self.get_idx(key) is len(self.keys())-1:
            return True
        else:
            return False
    
    def key_is_first(self,key):
        if self.get_idx(key) is 0:
            return True
        else:
            return False

    def get_idx(self,key,return_first=True):
        matches=[]
        for i,k in enumerate(self.keys()):
            if k==key:
                matches.append(i)
        if return_first:
            if len(matches)>1:
                print 'warning, multiple neurons with that name'
            return matches[0]
        else:
            return matches

def parse_stimulus_info(string):
    p1=string.find('(')
    p2=string.find(')')    
    start=int(string[:p1])
    finish=int(string[p1+1:p2])+start
    return np.array([start,finish])

from ConfigParser import ConfigParser
class WormConfig(ConfigParser):

    _ini_file=''
    def set_ini_file(self,f):###Something about inheriting ConfigParser messes up properties, need manual setter
        self._ini_file=f
        self.read(self._ini_file)
        self.get_random_details()

    def __init__(self,ini_file=None):
        ConfigParser.__init__(self)
        if ini_file is not None:
            self.set_ini_file(ini_file)
        
#        self.read(ini_file)
#        self.get_random_details()


    def get_random_details(self):
        self.xvox_nm=np.double(self.get('scaled image info','xvox_nm'))
        self.yvox_nm=np.double(self.get('scaled image info','yvox_nm'))
        self.zvox_nm=np.double(self.get('scaled image info','zvox_nm'))
        self.nm_voxel_shape=np.array([self.zvox_nm,self.yvox_nm,self.xvox_nm])
        self.zps=int(self.get('scaled image info','zps'))
        s=[self.zvox_nm,self.yvox_nm,self.xvox_nm]
        self.spacing=s/np.double(min(s))
        
        self.um_image_shape=np.array([
            np.double(self.get('scaled image info','zfovs_um')),##careful with order
            np.double(self.get('scaled image info','yfovs_um')),
            np.double(self.get('scaled image info','xfovs_um'))]).astype(np.double)
        
        self.nm_image_shape=self.um_image_shape*1000
        
        self.timepoints_per_second=np.double(self.get('raw imaging condition','img_speed_VPS'))#volumes per second
        try:
            self._stimulus_info=self.get('raw imaging condition','stimulus_timing')
            self.stimulus_seconds=parse_stimulus_info(self._stimulus_info)
            self.stimulus_timepoints=self.stimulus_seconds*self.timepoints_per_second
        except:
            self.stimulus_seconds=[]
            self.stimulus_timepoints=[]
            

        
def system_dependent_directories():
    if os.name is 'nt':#windows
        ki_dir='C:/Users/cgs567/Documents/Corral/Ki-Confocal-2015/'
        snyder_dir='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/'
    elif os.name is 'posix':#stampede
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



class tEnvironment(WormConfig,HasTraits):
    read_dir=Str
    write_dir=Str
    def __init__(self,read_dir=None,write_dir=None):
        WormConfig.__init__(self)
        if read_dir is not None:
            self.read_dir=read_dir
            if write_dir is None:
                self.write_dir=self.read_dir
        if write_dir is not None:
            self.write_dir=write_dir
        

    def _read_dir_changed(self,old,new):#Directories should end in / by my convention
        if self.read_dir[-1] != '/':
            self.read_dir+='/'
        #put in this loop so it doesn't trigger twice
        if len(old)==0:
            if self.get_ini_file():
                print 'Env found ini_file in', self.read_dir ,'...'
                print 'reading parameters...'

                WormConfig.__init__(self,self.ini_file)
#            tEnvironment.__init__(self,arg.read_dir,arg.write_dir)
#            self.config=WormConfig(self.ini_file)
            
    def _write_dir_changed(self):
        if self.write_dir[-1] != '/':
            self.write_dir+='/'
    def is_ini(self,f):
        try:
            return f[-4:]=='.ini'
        except:
            return False
    def get_ini_file(self):
        N=3;n=0
        read_dir_n=self.read_dir
        while n<N:
            self.lvln=[ os.path.join(read_dir_n,f) for f in os.listdir(read_dir_n)]
            ln=[f for f in self.lvln if self.is_ini(f)]
            if len(ln) != 0:
                self.ini_file=ln[0]
                break
            n+=1
            read_dir_n+='../'
        if n==N:
            print 'Was not able to find .ini file'
            return False
        else:
            return True


            
from skimage.io import imread as sk_imread
def add_leading_dimension2(x):
    return x[None,None, ...]
def add_leading_dimension(x):
    return x[None, ...]
def tokenize(*args):
    from hashlib import md5
    return md5(str(args).encode()).hexdigest()
from glob import glob


def simple_get_file_regex(folder):
    from ioworm import ls
    file_list=[f for f in ls(folder,fullpath=False) if '.tif' in f]
    try:
        sample_tif_file=file_list[0]
    except:
        raise IOError('No tif files in'+folder)
    p=sample_tif_file.rfind('.')
    regex='*****'
    re_file=sample_tif_file[0:p-len(regex)]+regex+sample_tif_file[p:]
    return re_file



###Someday will run into trouble with both npX and daX being defined differently
class tWorm(tEnvironment):
    stack=None
    re_file=None
#    spacing=Instance(Array,([1.0,1.0,1.0]))
#    stack_path=Str
    def __init__(self,*args):
        self.spacing=np.array([1.0,1.0,1.0])#default
#        self._use_precomputed_numpy=False
        tEnvironment.__init__(self)
        for arg in args:
            if isinstance(arg,tEnvironment):#environment class
                self.read_dir=arg.read_dir
                self.write_dir=arg.write_dir
#                tEnvironment.__init__(self,arg.read_dir,arg.write_dir)
            if isinstance(arg,str):#read_dir
                if os.path.isfile(arg):
                    if self.is_ini(arg):
                        self.set_ini_file(arg)
                else:
                    self.read_dir=arg
#                tEnvironment.__init__(self,arg)
            if isinstance(arg,np.ndarray):#numpy stack
                if np.size(arg)==3:
                    self.spacing=arg
                else:
                    self.npX=arg
            if hasattr(arg,'dask'):#dask array
                self.daX=arg

#        self.segments=Segments()

    def __getattr__(self,attr):
        if attr=='X':
#            print 'attr is X'
            try:
#                print 'i tried numpy first'
                return getattr(self,'npX')
            except:
#                print 'except i failed..returning dask array'
                try:
                    return getattr(self,'daX')
                except:
                    AttributeError('worm instance has no attribute npX or daX')
        else:
            try:
                return getattr(self,attr)
            except:
                raise AttributeError('Worm object has no attribute %r' % attr)

    def __setattr__(self,attr,value):
        if attr=='X':
            if isinstance(value,np.ndarray):
#                setattr(self,'npX',value)
                super(tWorm,self).__setattr__('npX',value)
            elif hasattr(value,'dask'):
                super(tWorm,self).__setattr__('daX',value)
            else:
                raise AttributeError('X must be an np.ndarray or dask.array')
        else:
            super(tWorm,self).__setattr__(attr,value)
            
                
            
    def __getitem__(self,*args):
        dstdask=self.X
        for a in args:
            dstdask=dstdask[a]
        return tWorm(tEnvironment(self.read_dir,self.write_dir),dstdask) #time or z slice of whole worm

    def get_numpy_patch(self,*args):
        dstdask=self.X        
        for a in args:
            dstdask=dstdask[a]
        return np.array(dstdask)

    def dask_read_data(self,re_file='*',n_slices=None):
        if (n_slices is None) and not hasattr(self,'zps'):#Must divide exactly for now
            print 'Error: no way to know how many planes per stack'
            return
        elif n_slices is None:
            n_slices=self.zps
            
        if re_file is '*':
            self.re_file=simple_get_file_regex(self.read_dir)
        else:
            self.re_file=re_file
        
        
        if hasattr(self,'read_dir'):
            self.re_file=self.read_dir+self.re_file
        else:
            print 'Error: no read_dir yet';return
        
        
        from movie_analysis import get_dask_array
        self.daX=get_dask_array(self.re_file,n_slices)##t,z,y,x
        
    

    def save(self,name):
#        if isinstance(self.Stack,np.ndarray):
        if name[-4:] != '.tif':
            name+='.tif'
        io.imsave(self.write_dir+name,self.daX.compute()) #assume name ends in .tif
#        io.imsave(self.write_dir+name,self.Stack) #assume name ends in .tif

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

    def display_on_scene(self,scene,downsample=1,t=0):#t is for default time point
        from mayavi.sources.api import ArraySource
        from mayavi.modules.volume import Volume
        self.array_src=ArraySource(spacing=self.spacing)
        
        f=downsample#take every fth element #just in x,y planes
        
        #####UNDER CONSTRUCTION#######
        if self.X.ndim is 4:
            dst=self.X[t]
        else:
            dst=self.X
        
        self.array_src.scalar_data=np.array(dst[:,::f,::f])#z,y,x

        scene.mayavi_scene.add_child(self.array_src)
        vol_module=Volume()
        self.volume=self.array_src.add_module(vol_module)



