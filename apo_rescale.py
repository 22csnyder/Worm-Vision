# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 02:07:12 2015

@author: cgs567
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 18:37:57 2014

@author: melocaladmin
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from traits.api import HasTraits

from traits.api import File

from WormBox.BaseClasses import tEnvironment as Environment

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




class Apo(HasTraits):
    file_name=File()
    full_marker_file=File()
    def __init__(self,env=None):
        if env is not None:
            self.env=env
    
    def _file_name_changed(self):
        if self.file_name[:6]=='/work/' or self.file_name[:3]=='C:/':##file_name is already fullpath
            self.marker_file=self.file_name
        else:
            self.marker_file=self.env.read_dir+self.file_name
        with open(self.marker_file,'r') as f:
            file_extension=self.marker_file.split('.')[-1]
            raw_list=[[get_number(line),parse_line(line,file_extension)] for line in f if line[0]!='#']#in x, y, z
            self.markers=dict()
            for label,marker in raw_list:
                self.markers[label]=marker



    def save_as_apo(self,name=None):
        if name is not None:
            self.save_name=name
        if not hasattr(self,'save_name'):self.save_name='unnamed_atlas.apo'
        self.full_save_name=self.env.write_dir+self.save_name
        with open(self.full_save_name,'w') as f:
            cnt=0
            for name,loc in self.markers.iteritems():
                xx,yy,zz=map(int,loc)
                if name.isdigit():
                    m=int(name)
                else:
                    m=cnt
                cnt=max(cnt,m+1)
                v3d_line=str(m)+',  '+str(m)+', '+name+', , '+str(zz)+','+str(xx)+','+str(yy)+', '+'0.000,0.000,0.000,50.000,0.000,,,,255,0,0\n'#red
#                v3d_line=str(m)+',  '+str(m)+', '+name+', , '+str(zz)+','+str(xx)+','+str(yy)+', '+'0.000,0.000,0.000,50.000,0.000,,,,0,255,0\n'#green
#                v3d_line=str(m)+',  '+str(m)+', '+name+', , '+str(zz)+','+str(xx)+','+str(yy)+', '+'0.000,0.000,0.000,50.000,0.000,,,,0,0,255\n'#blue
                f.write(v3d_line)


    def flip_z(self,n_slices):
        for name in self.markers.keys():
            self.markers[name][2]=n_slices-self.markers[name][2]
    
    def rescale(self,offset,dialation):
        for dim in range(len(offset)):
            b=offset[dim];a=dialation[dim]
            for name in self.markers.keys():
                i=self.markers[name][dim]
                self.markers[name][dim]=np.round(  a*(i-b) )



#data_folder='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150903_CM_adult_agar_pad_W1/'
#apo_file='150902_1703_CM_W1_cropped at 388-714y_80-552x.apo'
#env=Environment(data_folder)
#apo=Apo(env)
#apo.file_name=apo_file
#apo.flip_z(101)
#apo.rescale([0,0,16],[1/2.0,1/2.0,1/3.3])
#apo.save_as_apo('flipped.apo')


#from WormBox.BaseClasses import tEnvironment as Env
#from WormBox.BaseClasses import WormConfig
#old_ini='C:/Users/cgs567/Documents/Corral/Ki-Confocal-2015/151001_CM_Glycerol_3ch_Dev_II_W3/bin_info.ini'
#new_ini='C:/Users/cgs567/Documents/Corral/Ki-Confocal-2015/151001_2012_Vol_Img_Glycerol_3ch_Dev_II_W3/Vol_Img_Info.ini'
#old_wcfig=WormConfig(old_ini)
#new_wcfig=WormConfig(new_ini)
#scaling=old_wcfig.voxel_shape / new_wcfig.voxel_shape
#data_folder='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/151001_CM_Glycerol_3ch_Dev_II_W3/'
#apo_file='151001_HQ1_Gly_3ch_D_II_W3_L4_O20X.apo'
#apo=Apo(Env(data_folder))
#apo.file_name=data_folder+apo_file
#apo.flip_z(61)
#apo.rescale([0,0,0],scaling)
#apo.save_as_apo('flipped.apo')


from WormBox.BaseClasses import tEnvironment as Env
from WormBox.BaseClasses import WormConfig
#old_ini='C:/Users/cgs567/Documents/Corral/Ki-Confocal-2015/151001_CM_Glycerol_3ch_Dev_II_W3/bin_info.ini'
#new_ini='C:/Users/cgs567/Documents/Corral/Ki-Confocal-2015/151001_2012_Vol_Img_Glycerol_3ch_Dev_II_W3/Vol_Img_Info.ini'
#old_wcfig=WormConfig(old_ini)
#new_wcfig=WormConfig(new_ini)
#scaling=old_wcfig.voxel_shape / new_wcfig.voxel_shape
#data_folder='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/151001_CM_Glycerol_3ch_Dev_II_W3/'
#apo_file='151001_HQ1_Gly_3ch_D_II_W3_L4_O20X.apo'
#apo=Apo(Env(data_folder))
#apo.file_name=data_folder+apo_file
#apo.flip_z(61)
#apo.rescale([0,0,0],scaling)
#apo.save_as_apo('flipped.apo')


old_ini='C:/Users/cgs567/Documents/Corral/Ki-Confocal-2015/150902_CM_Dev_II_No_Stimulus_W1/TIFF_info.ini'
new_ini='C:/Users/cgs567/Documents/Corral/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/TIFF.ini'
old_wcfig=WormConfig(old_ini)
new_wcfig=WormConfig(new_ini)
data_folder='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/'
apo=Apo(Env(data_folder))
apo.file_name='150902_1703_Vol_W1_t151012chris.apo'
apo.flip_z(old_wcfig.zps)
scaling=old_wcfig.voxel_shape / new_wcfig.voxel_shape
apo.rescale([0,0,18],scaling)
apo.save_as_apo('t151012 flipped.apo')


zmin=np.inf;zmax=-np.inf
for m in apo.markers.values():
    zval=m[2]
    if zval>zmax:
        zmax=zval
    if zval<zmin:
        zmin=zval

print 'zmin is ',zmin
print 'zmax is ',zmax




#apo2=Apo(Environment('C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/'))
#apo2.file_name='150902_1703_Vol_W1_cropped at 194-357y_40-276x 150908chris.apo'
#apo2.save_as_apo()





#def is_sorted(L):
#    for i in range(len(L)-1):
#        if L[i]>L[i+1]:
#            return False
#    return True
#
#
#
#_x=_loc[0];_y=_loc[1];_z=_loc[2]
#
#
#ans=sorted(zip(_names,_x,_y,_z),key=lambda tup:tup[2])
#names,x,y,z=zip(*ans)




