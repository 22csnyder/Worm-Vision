# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:44:23 2015

@author: csnyder
"""
import numpy as np


results_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151112_NaCl_10mM_30s_no_anes_W2/Results/'



import dill
with open(results_dir+'t151123_Intensity.pkl','rb') as handle:
    Intensity=dill.load(handle)
with open(results_dir+'t151123_chord_only.pkl','rb') as handle:
    segments=dill.load(handle)

import matplotlib.pyplot as plt

from produce_correlation_plots import plot_correlation_matrix,dict2mat
from WormBox.WormPlot import getsquarebysideaxes

_Data,_Labels=dict2mat(Intensity)
x_pos=np.array([segments[name]['marker'][2] for name in _Labels])
inds=x_pos.argsort()
Data=_Data[inds];Labels=_Labels[inds]


#plot_correlation_matrix(Data,Labels)
#plt.title('Volume Intensity over Time 151112 NaCl W2')
#plt.show()

VPS=1.94
F0=Data[:,0][:,np.newaxis]
sF=100*(Data-F0)/F0

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
#plt.matshow(sF,aspect=3)
#cbar=plt.colorbar()
#cbar.set_clim(-50,100)
#cbar.draw_all()
#plt.title('delF/F0')
#plt.yticks(np.arange(len(Labels))+0.5,Labels)
#plt.tick_params(axis='x',labeltop='off',labelbottom='on')
#increment=5#s
#total_seconds=229/VPS#118seconds total=229/1.94
#seconds=np.mgrid[0:total_seconds:increment]
#tick_location=[t*VPS for t in seconds]
#plt.xticks( tick_location,seconds.astype(np.int))
#plt.xlabel('time (s) StimOn=30s StimOff=60s')
#plt.ylabel('neuron label')

####PLOT ALL ON ONE CROWDEDLY####
#time=np.arange(229)/VPS
#plt.plot(time,sF.transpose())



import scipy
fft=np.abs(scipy.fft(sF))
t=range(229)
L=49
s0=0
s1=len(t)
fig,axes=getsquarebysideaxes(L)
for i in range(L):
    ax=axes[i]
    label=Labels[i]
    y=fft[i]
    ax.plot(t[s0:s1],y[s0:s1],'b')
    ax.set_title(label)
    ax.set_xlim(s0,s1)
    ax.set_ylim(0,500)
plt.show()
plt.tight_layout(pad=0.5,w_pad=0.5,h_pad=0.6)



t=np.linspace(0,229)
T=t[-1]

freq=8
s=np.sin(2*np.pi*freq*t/T)
plt.plot(t,s)

f1=scipy.fft(s)




