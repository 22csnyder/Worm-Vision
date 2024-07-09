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
    CM=plt.pcolor(np.flipud(R))#pass cmap as arg here if nec
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
    
    return Data,Labels
        


def plot_intensity_correlation_matrix(Intensity):
    Data,Labels=dict2mat(Intensity) 
    plot_correlation_matrix(Data)



#Data,Labels=dict2mat(Intensity)
#
#D1=Data[:,:50]
#D2=Data[:,50:100]
#D3=Data[:,100:150]
#D4=Data[:,150:]
#
#plot_correlation_matrix(D4,Labels);plt.show()



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
plot_correlation_matrix(D+2,Labes)
plt.figure()
plot_correlation_matrix(D,Labes)
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

