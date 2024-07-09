# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:28:04 2015

@author: csnyder
"""
import time
import numpy as np
import os
import dill
from traits.api import Instance,HasTraits,Str
from NeuronTrace import TracePlot
from WormBox.BaseClasses import Segments, DEFAULT_KEY
import matplotlib.pyplot as plt

from produce_correlation_plots import plot_correlation_matrix,dict2mat

from WormBox.WormPlot import getsquarebysideaxes


def str2int(src):
    def f(string):
        return np.int(np.float(string))
    
    if isinstance(src,list):
        return np.array(map(f,src))
    else:
        return f(src)
        


####################################################################################################
class NeuronAnalyzer(HasTraits):
    
    trace=Instance(TracePlot,())    
    intensity=dict()
    position=dict()
    
    segments=None
#    view = View(Item('figure', editor=MPLFigureEditor(),
#                           show_label=False),
#               Item('n'),
#               Item('a'),
#                   width=400,
#                   height=300,
#                   resizable=True)
        
    segments_file=Str('')
    intensity_file=Str('')
    save_dir=Str('')
    
    VPS=None
    stimulus_timepoints=np.empty((0))
    F0=None#Baseline intensity
        
    
    def _segments_file_changed(self):
        #Read Mode        
        if os.path.isfile(self.segments_file):
            self.load_segments(self.segments_file)
        #Write Mode
        else:
            if self.segments is None:
                print 'Warning: segments_file is not an existing file'
            else:
                self.save_segments(self.segments_file)
    def _intensity_file_changed(self):
        #Read Mode        
        if os.path.isfile(self.intensity_file):
            self.load_intensity(self.intensity_file)
        #Write Mode
        else:
            if self.intensity is None:
                print 'Warning: intensity_file is not an existing file'
            else:
                self.save_intensity(self.intensity_file)                
    
 
    
    def __init__(self):
        pass

    
    def _init_from_gui(self,segments,vol,save_dir=''):
        self.segments=segments
        self.save_dir=save_dir
        self.vol=vol
        self.trace=TracePlot(vol.X.shape[0])
        self.stimulus_timepoints=self.vol.stimulus_timepoints
        self.trace.vertical_lines=self.vol.stimulus_timepoints
        self.VPS=self.vol.timepoints_per_second
        self.T=len(self.vol.X)#total number of timepoints
        
        self.nm_voxel_shape=self.segments.seg_channel.nm_voxel_shape
    def _save_dir_changed(self):
        if self.save_dir[-1]=='/':
            self.intensity_file=self.save_dir+'intensity.pkl'
        else:
            self.intensity_file=self.save_dir+'/'+'intensity.pkl'
        
#        self.segments_file=self.save_dir+'/'+'segments.pkl'
    def update_data_for(self,name):
        current_neuron=self.segments[name]
        if not current_neuron.has_key('mesh'):
            print 'warning ', name,' not segmented'
            return
        
        current_mesh=current_neuron['mesh']
        point=current_neuron['marker']// self.vol.nm_voxel_shape        
        
        I_seg=current_mesh.return_segmented_image()
        patch_data=current_mesh.unscaled_patch
        ix,iy,iz=np.where(I_seg==1)
        m=patch_data[I_seg]
        inds=m.argsort()
        
#        percent=40.0
#        interior_size=I_seg.sum()
#        topinds=np.round(interior_size*percent/100)
#        select_inds=inds[(interior_size-topinds) :] #This gets the ones with highest intensity
        
        select_inds=inds##just take all of them until I understand what I did
        bix=ix[select_inds]
        biy=iy[select_inds]
        biz=iz[select_inds]
        
        [x_pos,y_pos,z_pos]=current_neuron['marker']//self.vol.nm_voxel_shape  - self.vol.window_radius
        [x_len,y_len,z_len]=self.vol.window_shape
#        try:##Still bad at handling edge cases
        y=[]
        time_points=range(self.vol.X.shape[0])

        time_unscaled_patch=self.vol.get_window_sized_unscaled_padded_patch(point,time_step=time_points)

        s1=time.time()
        
        for t in time_points:
            unscaled_patch=time_unscaled_patch[t]
#            unscaled_patch=self.vol.get_window_sized_unscaled_padded_patch(point,time_step=t)
            score=np.mean(unscaled_patch[bix,biy,biz])
            y.append(score)
            
#        for t in time_points:
#            unscaled_patch=self.vol.get_window_sized_unscaled_padded_patch(point,time_step=t)
#            score=np.mean(unscaled_patch[bix,biy,biz])
#            y.append(score)

        self.intensity[name]=np.array(y)
        print 'intensity update total time: ',time.time()-s1        
        
#            return True
#        except:
#            print 'Warning, edge cases not well handled'
#            return False

    def update_plot_for(self,name):
        self.trace.y=self.intensity[name]
        self.trace.update_title(name)
        self.trace.update_plot()

    def load_intensity(self,f=None):
        print 'Reading file: ',self.intensity_file
        if f is None:
            f=self.intensity_file
        with open(f,'rb') as handle:
            self.intensity=dill.load(handle)

    def save_intensity(self,f=None):
        print 'Saving as file: ',self.intensity_file
        if f is None:
            f=self.intensity_file
        with open(f,'wb') as handle:
            dill.dump(self.intensity,handle)

    def load_segments(self,f=None):
        print 'Reading file: ',self.segments_file
        with open(f,'rb') as handle:
            self.segments=dill.load(handle)
#
#    def save_segments(self,f=None):
#        print 'Saving as file: ',self.segments_file
#        with open(f,'wb') as handle:
#            dill.dump(self.segments,handle)

    #Warning, this should take a long time because it's not parrallel and 
    #it iterates through all neurons
    def calculate_all_intensities_from_segments(self):
        for name in self.segments.keys():
            self.update_data_for(name)
            
    def _prelim_analysis(self):
        self._Data,self._Labels=dict2mat(self.intensity)
        if self.segments is not None:
            self.x_pos=np.array([self.segments[name]['marker'][2] for name in self._Labels])
        else:
            self.x_pos=np.array([self.position[name][2] for name in self._Labels])
        inds=self.x_pos.argsort()
        self._Data=self._Data[inds];self._Labels=self._Labels[inds]

    def plot_correlation(self,title=None):
            plot_correlation_matrix(self._Data,self._Labels)
            if title is not None:
                plt.title(title)

#    def plot_fancy_time_series(self,title=None):
#        
#        
##plt.matshow(sF,aspect=3)
##cbar=plt.colorbar()
##cbar.set_clim(-50,100)
##cbar.draw_all()
##plt.title('delF/F0')
##plt.yticks(np.arange(len(Labels))+0.5,Labels)
##plt.tick_params(axis='x',labeltop='off',labelbottom='on')
##increment=5#s
##total_seconds=229/VPS#118seconds total=229/1.94
##seconds=np.mgrid[0:total_seconds:increment]
##tick_location=[t*VPS for t in seconds]
##plt.xticks( tick_location,seconds.astype(np.int))
##plt.xlabel('time (s) StimOn=30s StimOff=60s')
##plt.ylabel('neuron label')


    def plot_time_series(self,title=None,names=None):
        if names is None:
            keys=self.intensity.keys()
        else:
            keys=map(str,names)
        T=len(self.intensity.values()[0])
        t=range(T)
        fig,axes=getsquarebysideaxes(len(keys))
        for key,ax in zip(self._Labels,axes):
            y=self.intensity[key]
            ax.plot(t,y,'b')
            ax.set_title(key)
            ax.set_xlim(0,T)
            for x in self.stimulus_timepoints:
                ax.axvline(x,color='r',linestyle='--')
        if title is not None:
            plt.suptitle(title)
        plt.tight_layout(pad=0.5,w_pad=0.5,h_pad=0.6)
        plt.show()


    def write_csv(self):
        self.write_intensity_csv()
        self.write_pixel_position_csv()
        self.write_nm_position_csv()

    def write_intensity_csv(self,name=None):
        import csv
        folder=self.save_dir
        if name is None:
            name='intensity.csv'
        f=open(folder+'/'+name, 'wb')
        c = csv.writer(f)
        for label,datum in zip(self._Labels,self._Data):
            c.writerow( [label]  +  list(datum) )
        f.close()

    def write_pixel_position_csv(self,name='neuron_pixel_position.csv'):
        import csv
        folder=self.save_dir
        f=open(folder+'/'+name, 'wb')
        c = csv.writer(f)
        c.writerow(['label','z (pixels)','y (pixels)','x (pixels)'])
        for label,neuron in zip(self._Labels,self.segments.values()):
            if 'marker' not in neuron.keys():
                continue
            pos=np.array(neuron['marker']/self.nm_voxel_shape,np.int).astype(np.int)
            c.writerow( [label]  +  list(pos) )
        f.close()

    def write_nm_position_csv(self,name='neuron_nm_position.csv'):
        import csv
        folder=self.save_dir
        f=open(folder+'/'+name, 'wb')
        c = csv.writer(f)
        c.writerow(['label','z (nm)','y (nm)','x (nm)'])
        for label,neuron in zip(self._Labels,self.segments.values()):
            if 'marker' not in neuron.keys():
                continue
            c.writerow( [label]  +  list(neuron['marker']) )
        f.close()

    def read_nm_position_csv(self,name):
        import csv
        with open(name,'r') as handle:
            iter_pos=csv.reader(handle)
            iter_pos.next()
            for line in iter_pos:
                name=line[0]
                pos= str2int(line[1:])
                self.position[name]=pos

#s0=0
#s1=60
#L=49
#t=range(len(Intensity.values()[0]))
#fig,axes=getsquarebysideaxes(L)
#for i in range(L):
#    ax=axes[i]
#    label=Labels[i]
#    y=Data[i]
#    ax.plot(t[s0:s1],y[s0:s1],'b')
#    ax.set_title(label)
#    ax.set_xlim(s0,s1)
#plt.show()
#plt.tight_layout(pad=0.5,w_pad=0.5,h_pad=0.6)



######PLOT FANCY TIME SERIES TABLE######



    def plot_fancy_time_series(self):
        self.F0=self._Data[:,0][:,np.newaxis]
        self.sF=100*(self._Data-self.F0)/self.F0
        ######PLOT FANCY TIME SERIES TABLE######
        plt.matshow(self.sF,aspect=3)
        cbar=plt.colorbar()
        cbar.set_clim(-50,100)
        cbar.draw_all()
        plt.title('delF/F0')
        plt.yticks(np.arange(len(self._Labels))+0.5,self._Labels)
        plt.tick_params(axis='x',labeltop='off',labelbottom='on')
        increment=15#s
        total_seconds=self.T/self.VPS#118seconds total=229/1.94
        seconds=np.mgrid[0:total_seconds:increment]
        tick_location=[t*self.VPS for t in seconds]
        plt.xticks( tick_location,seconds.astype(np.int))
        plt.xlabel('time (s)')
        plt.ylabel('neuron label')

####PLOT ALL ON ONE CROWDEDLY####
#time=np.arange(229)/VPS
#plt.plot(time,sF.transpose())


#import scipy
#fft=np.abs(scipy.fft(sF))
#t=range(229)
#L=49
#s0=0
#s1=len(t)
#fig,axes=getsquarebysideaxes(L)
#for i in range(L):
#    ax=axes[i]
#    label=Labels[i]
#    y=fft[i]
#    ax.plot(t[s0:s1],y[s0:s1],'b')
#    ax.set_title(label)
#    ax.set_xlim(s0,s1)
#    ax.set_ylim(0,500)
#plt.show()
#plt.tight_layout(pad=0.5,w_pad=0.5,h_pad=0.6)
#
#
#
#t=np.linspace(0,229)
#T=t[-1]
#
#freq=8
#s=np.sin(2*np.pi*freq*t/T)
#plt.plot(t,s)
#
#f1=scipy.fft(s)




def write_csv(labels,data,folder,name):
    import csv
    f=open(folder+'/'+name, 'wb')
    c = csv.writer(f)
    for label,datum in zip(labels,data):
        c.writerow( [label]  +  list(datum) )
    f.close()





if __name__ == '__main__':
    analyzer=NeuronAnalyzer()
    
    #read data
    analyzer.intensity_file= '/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Results/Intensity.pkl'
    analyzer.segments_file='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Results/Segments.pkl'

    analyzer.stimulus_timepoints=np.array([ 232.8, 291. ])
    analyzer.VPS=1.94

    analyzer._prelim_analysis()
#    preset_title='151125 W2 10mM NaCl 120(30) 4min'
#    analyzer.plot_correlation(preset_title)
#    analyzer.plot_time_series(preset_title)




    #F0=Data[:,0][:,np.newaxis]
    #sF=100*(Data-F0)/F0

    folder='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Results/'
    ##Save as csv
    name='151125_intensity.csv'
    labels=analyzer._Labels
    data=analyzer._Data


    write_csv( labels, data, folder, name)


    name='151125_locations.csv'
    labels=analyzer._Labels
    
    
    data=np.vstack([analyzer.segments[l]['marker'] for l in labels])


    write_csv( labels, data, folder, name)
    
    
    
    


