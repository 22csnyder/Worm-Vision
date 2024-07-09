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
    


def plot_correlation_matrix(Intensity):
    Labels=[];data_list=[]
    for key in Intensity.keys():
        Labels.append(key)
        data_list.append(Intensity[key])
        
    Data=np.vstack(data_list)
    
    
    ## generating some uncorrelated data
    #Data = np.random.rand(10,100) # each row of represents a variable
    #Labels=['a','b','c','d','e','f','g','h','i','j']
    
    # plotting the correlation matrix
    R = np.corrcoef(Data)
    
    CM=plt.pcolor(np.flipud(R))#pass cmap as arg here if nec
    #
    plt.xlim(0,len(Labels))
    plt.ylim(0,len(Labels))
    #plt.yticks(np.arange(0.5,10.5),range(0,10))
    plt.xticks(np.arange(len(Labels))+0.5,Labels,rotation=90)
    plt.yticks(np.arange(len(Labels))+0.5,Labels[::-1])
    plt.xlabel('Neuron ID')
    plt.ylabel('Neuron ID')
    plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
    
    ax=plt.gca()
    ax.yaxis.set_label_position('right')
    
    plt.colorbar(CM, pad=0.07)#pad shifts it to the right for when label is to the right
    
    plt.show()

plot_correlation_matrix(Intensity)



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

