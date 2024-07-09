# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 09:09:12 2015

@author: csnyder
"""
write_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/Seg1/'

import numpy as np
import dill
import matplotlib.pyplot as plt

with open(write_dir+'/t150916 IntensityTrace.pkl','r') as handle:
    Intensity=dill.load(handle)
    

def plot_correlation_matrix(Data,Labels=None):
    ## generating some uncorrelated data
    #Data = np.random.rand(10,100) # each row of represents a variable
    #Labels=['a','b','c','d','e','f','g','h','i','j']
    
    # plotting the correlation matrix
    R = np.corrcoef(Data)
#    CM=plt.pcolor(np.flipud(R))#pass cmap as arg here if nec
    CM=plt.pcolor(np.flipud(R),vmin=-1,vmax=1)#pass cmap as arg here if nec
    n=Data.shape[0]
    plt.xlim(0,n)
    plt.ylim(0,n)

    if Labels is not None:
        plt.xticks(np.arange(len(Labels))+0.5,Labels,rotation=90)
        plt.yticks(np.arange(len(Labels))+0.5,Labels[::-1])
        
    #plt.yticks(np.arange(0.5,10.5),range(0,10))
    plt.xlabel('Neuron ID')
    plt.ylabel('Neuron ID')
    plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
    ax=plt.gca()
    ax.yaxis.set_label_position('right')
    plt.colorbar(CM, pad=0.07)#pad shifts it to the right for when label is to the right
#    plt.show(block=False)

def dict2mat(Intensity):
    Labels=[];data_list=[]
    for key in Intensity.keys():
        Labels.append(key)
        data_list.append(Intensity[key])
    Data=np.vstack(data_list)
    return Data,np.array(Labels)
        


def plot_intensity_correlation_matrix(Intensity):
    Data,Labels=dict2mat(Intensity)
    plot_correlation_matrix(Data)

if __name__=='__main__':
    #Data,Labels=dict2mat(Intensity)
    #
    #D1=Data[:,:50]
    #D2=Data[:,50:100]
    #D3=Data[:,100:150]
    #D4=Data[:,150:]
    #
    #plot_correlation_matrix(D4,Labels);plt.show()
    
    
    #####Redo NoStimData#####
    mean_pkl='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/Seg1/t151014MeanIntensity.pkl'
    top25_pkl='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/Seg1/t151014MedianIntensity.pkl'
    lower25_pkl='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/Seg1/t151014Lower75Intensity.pkl'
    seg_pkl='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/Seg1/t151014segments_dict.pkl'
    import dill
    with open(mean_pkl,'r') as handle:
        uIntensity=dill.load(handle)
    with open(top25_pkl,'r') as handle:
        qIntensity=dill.load(handle)#q for top quartile
    with open(lower25_pkl,'r') as handle:
        lowIntensity=dill.load(handle)
    with open(seg_pkl,'r') as handle:
        segments=dill.load(handle)
        
    
    Intensity=uIntensity
    title_prefix='Mean'
    
    Intensity=qIntensity
    title_prefix='Brightest 25% pixels in'
    
    Intensity=lowIntensity
    title_prefix='Dimmest 25% pixels in'
    
    
    _Data,_Labels=dict2mat(Intensity)
    
    
    x_pos=np.array([segments[name]['marker'][0] for name in _Labels])
    inds=x_pos.argsort()
    
    
    #int_Labels=_Labels.astype(np.int)
    #inds=int_Labels.argsort()
    
    Data=_Data[inds];Labels=_Labels[inds]
    
    
    
    plot_correlation_matrix(Data,Labels)
    plt.title(title_prefix+ ' Volume Intensity over Time NoStimulus150902_1703')
    
    plt.figure()
    shortData=Data[:,0:50]
    plot_correlation_matrix(shortData,Labels)
    plt.title(title_prefix+' Volume Intensity over Time (0-25s) NoStimulus150902_1703')
    
    plt.figure()
    shortData=Data[:,50:100]
    plot_correlation_matrix(shortData,Labels)
    plt.title(title_prefix+' Volume Intensity over Time (25-50s) NoStimulus150902_1703')
    
    plt.figure()
    shortData=Data[:,100:150]
    plot_correlation_matrix(shortData,Labels)
    plt.title(title_prefix+' Volume Intensity over Time (50-75s) NoStimulus150902_1703')
    
    plt.figure()
    shortData=Data[:,150:]
    plot_correlation_matrix(shortData,Labels)
    plt.title(title_prefix+' Volume Intensity over Time (75s-116s) NoStimulus150902_1703')
    
    plt.show()
    
    #How many are bad
    bad_so_far=0
    for name in segments.keys():
        if 'notes' in segments[name].keys():
            if 'bad' in segments[name]['notes']['quality']:
                bad_so_far+=1
                

    
    H=np.mean(Data,axis=1)
    plt.hist(H,bins=20)
    plt.title(title_prefix+' Volume Intensity--time average')
    plt.show()
    
    
    
    
    ##RANDOM SAMPLES FROM NO_STIM DATA###
    #N=27
    #Labes=[str(a) for a in range(N)]
    #from WormBox.BaseClasses import tWorm as Worm
    #no_stim_read_dir='/work/03176/csnyder/Corral/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/TIFF/'
    #worm=Worm(no_stim_read_dir)
    #worm.dask_read_data()
    #shape=worm.daX.shape
    #idx=(np.random.randint(0,shape[1],N)
    #    ,np.random.randint(0,shape[2],N)
    #    ,np.random.randint(0,shape[3],N))
    #traces=[]
    #for i in range(N):
    #    traces.append(worm.daX[:,idx[0][i],idx[1][i],idx[2][i]].compute())
    #Data=np.vstack(traces)#Dask doesn't yet support fancy indexing 
    #plot_correlation_matrix(Data,Labes)
    #plt.show()
    
    
    ###RANDOM MATRIX###
    N=27;T=232;
    D=np.random.rand(N,T)
    Labes=[str(a) for a in range(N)]
    #D=np.vstack([a,a,a,a])
    plt.figure()
    plot_correlation_matrix(D,Labes)
    plt.title('Representative "Random" Correlation matrix T=232 N=27')
    plt.show()
    #########
    N=81;T=232;
    D=np.random.rand(N,T)
    Labes=[str(a) for a in range(N)]
    #D=np.vstack([a,a,a,a])
    plt.figure()
    plot_correlation_matrix(D,Labes)
    plt.title('Representative "Random" Correlation matrix T=232 N=27')
    plt.show()
    
    
    #if __name__=='__main__':
    #    plot_correlation_matrix(Intensity)
    
    
    
    
    #ax=plt.subplot()
    #ConfusionMatrix=ax.pcolor(np.flipud(R))
    #cbar=plt.colorbar(ConfusionMatrix,cax=ax)
    ##ax.yticks(np.arange(0.5,10.5),range(0,10))
    #ax.set_xticklabels(Labels,rotation=90)
    #ax.set_yticklabels(Labels[::-1])
    ##ax.yticks(np.arange(len(Labels))+0.5,Labels[::-1])
    #ax.set_xlabel('Neuron ID')
    #ax.set_ylabel('Neuron ID')
    #ax.yaxis.set_label_position('right')
    ##ax.tick_params(axis='y', which='both', labelleft='off', labelright='on')
    ##ax.show()
    #plt.show()
    
