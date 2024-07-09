
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 15:15:37 2015

@author: melocaladmin
"""
import os
import shutil
import pims
import sys
import re
import numpy as np



def reorient(frame):
    return frame[::-1,...]

def getPIMWormData(filename =-1):
    if filename==-1:
#        print("does happen")
        macfoldername='/Users/christophersnyder/Documents/BenYakarLab/Confocal_Images/WormPics/'
        pcfoldername="C:\\Users\\melocaladmin\\My Documents\\WormPics"
        macdataname='/20140908 Confocal ZIM294 Step Size Stacks Split/14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack.tif'
        pcdataname="\\20140908 Confocal ZIM294 Step Size Stacks\\14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack.tif"
        print("hiya")
        if sys.platform=='darwin':
            datafile=macfoldername+macdataname
        elif sys.platform in ['linux','linux2']:#may be broken
#            linuxfoldername='/home/christopher/Documents/Ben-Yakar/WormPics'
#            ldata1='/home/christopher/Documents/Ben-Yakar/WormPics/20140908 Confocal ZIM294 Step Size Stacks Split'
#            ldata2=
#            datafile=linuxfoldername+macdataname
            datafile='/home/christopher/Documents/Ben-Yakar/WormPics/20140908 Confocal ZIM294 Step Size Stacks/14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack.tif'
            print("this happened")
        else:
            datafile=pcfoldername+pcdataname

        print(datafile)
        v=pims.TiffStack(datafile,process_func=reorient)
    else:
        v=pims.TiffStack(filename,process_func=reorient)
                
    return v
    

def print_indicies_to_file(L,filename):
    if L.__len__() in (2, 3):
        f=open(filename,'w')
        atlas=zip(*L)
        for pt in atlas:
            if pt.__len__()==2:
                f.write('%d %d \n' % pt)
            elif pt.__len__()==3:
                f.write('%d %d %d \n' % pt)
        f.close()
    else:
        print("warning haven't described what to print in this case")
        

path_name='RecentSegmentation'

def create_fresh_directory(path_name):
    if os.path.exists(path_name):
        shutil.rmtree(path_name)
    os.mkdir(path_name)

def time_stack_read():
    pcfoldername="C:\\Users\\melocaladmin\\Documents\\WormPics"
    time_stack_file="\\141226_CM_Pan_GCamp_Vol1_Stacks\\c_141226_ZIM294 Vol1_L4_O20X_F630x75um_Vol_P6144x500S28_g2_stack1of7.tiff"
    path=pcfoldername+time_stack_file
    
    m=getPIMWormData(path)
    return m
 
def clean_print(my_list):
    for m in my_list:
        print(m)
        print(' ')

           
def ls(d='default_cwd',fullpath=1,verbose=0):#currently returns full path no matter what
    if d=='default_cwd':d=os.getcwd()    
    if not os.path.exists(d):
        print("path is no good")
        return
    else:
#        print("d is ")
#        print(d)
        if fullpath==1:
            out=[os.path.join(d,f) for f in os.listdir(d)]
#            return os.listdir(d)
        else:
            out=os.listdir(d)
    if verbose:
        clean_print(out)
    return out
    

    
def return_those_ending_in(str_list,str_ending):
    file_type=[s.split('.',-1)[-1] for s in str_list]
    clean_list=[]    
    for f,s in zip(file_type,str_list):
        if f in[str_ending,'.'+str_ending]:
            clean_list.append(s) 
    return clean_list


def tryint(s):
    try:
        return int(s)
    except:
        return s
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]
def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def assert_int(s):
    try:
        return int(s)
    except:
        pass
    
def get_file_index(s):
    ank=alphanum_key(s)
    w=[a for a in ank if isinstance(a,int)]
    if len(w)==0:
        return ank[-1]
    else:
        return w[-1]
 
def easy_contents(readdir):
    short_files=ls(readdir,fullpath=True)
    
    ix=[get_file_index(s) for s in short_files]
    
    input_data=[[i,f] for i,f in zip(ix,short_files)]
    input_data.sort(key=lambda pair: pair[0])
    return zip(*input_data)
    
def save_list_of_arrays_in_a_folder(foldername,List):
    if isinstance(List,np.ndarray):
        List=[List]
    create_fresh_directory(foldername)#relative to current
    os.chdir(foldername)
    numlist=range(List.__len__())
    filenames=['array'+str(x) for x in numlist]
    
    for (f,mat) in zip(filenames,List):
        np.save(f,mat)
    os.chdir('..')

def convert_index(ix,n_digits):
    s=str(ix)
    while len(s)<n_digits:
        s='0'+s
    return s
    
#a=np.linspace(0,10,11)
#b=np.array([[3,5,2,11],[2,1,1,5]])
#pathWormPics='./../Corral/Snyder/WormPics'
#l=[b,b,b]
#
#save_list_of_arrays_in_a_folder('saveresultstest',l)

def load_list_of_arrays_in_a_folder(foldername):
    if not os.path.exists(foldername):
        print("your folder does not exist in the current directory")
    filelist=ls(foldername)
    return [np.load(f) for f in filelist]#The order does not seem to be preserved
        

def smakedir(f):
    import os
    if not os.path.exists(f):
        os.makedirs(f)

def smakedirs(*arg):
    if len(arg)==1:
        f=arg[0]
        smakedir(f)
    else:
        for f in arg:
            smakedir(f)
            
def _test(*arg):
    print len(arg)

        
        
        

