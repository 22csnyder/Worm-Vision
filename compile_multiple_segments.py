# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:45:19 2016

@author: csnyder
"""
import numpy as np

files=[
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos-3_248_233_len23_157_251/Results/intensity.pkl',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos-3_106_361_len23_173_191/Results/intensity.pkl',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_49_439_len17_107_203/Results/intensity.pkl',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_69_637_len17_107_203/Results/intensity.pkl',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_142_782_len17_223_203/Results/intensity.pkl',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_318_880_len17_151_271/Results/intensity.pkl',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_62_1079_len17_319_233/Results/intensity.pkl',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/FOV8/Results/intensity.pkl',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/FOV9/Results/intensity.pkl',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos-2_335_1722_len15_63_113/Results/intensity.pkl',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos-2_244_1853_len25_149_203/Results/intensity.pkl']

position_files=[
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos-3_248_233_len23_157_251/Results/neuron_nm_position.csv',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos-3_106_361_len23_173_191/Results/neuron_nm_position.csv',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_49_439_len17_107_203/Results/neuron_nm_position.csv',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_69_637_len17_107_203/Results/neuron_nm_position.csv',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_142_782_len17_223_203/Results/neuron_nm_position.csv',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_318_880_len17_151_271/Results/neuron_nm_position.csv',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_62_1079_len17_319_233/Results/neuron_nm_position.csv',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/FOV8/Results/neuron_nm_position.csv',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/FOV9/Results/neuron_nm_position.csv',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos-2_335_1722_len15_63_113/Results/neuron_nm_position.csv',
'/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos-2_244_1853_len25_149_203/Results/neuron_nm_position.csv']

def add_letter(let,arr):
    return np.array([let+a for a in arr])


alph=['a','b','c','d','e','f','g','h','i','j','k']

from NeuronAnalyzer import NeuronAnalyzer


data=[];labels=[]
for i,letter in enumerate(alph):
    na=NeuronAnalyzer()
    na.intensity_file=files[i]
    na.read_nm_position_csv(position_files[i])
    na._prelim_analysis()
    data.append(na._Data)
    labels.append(add_letter(letter,na._Labels))
    
    
a=NeuronAnalyzer()
a._Data=np.concatenate(data)
a._Labels=np.concatenate(labels)


d=a._Data
import scipy

R = np.corrcoef(d)

D=1-R


#a.plot_time_series()
a.plot_correlation()
#a.write_csv()
#a.plot_fancy_time_series()


#position_file='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_318_880_len17_151_271/Results/neuron_nm_position.csv'
#import csv
#f=open(position_files[0],'r')
#P=csv.reader(f)







import scipy
#import pylab
from matplotlib import pylab
import scipy.cluster.hierarchy as sch

## Generate features and distance matrix.
#x = scipy.rand(40)
#D = scipy.zeros([40,40])
#for i in range(40):
#    for j in range(40):
#        D[i,j] = abs(x[i] - x[j])


# Compute and plot dendrogram.
fig = pylab.figure()
axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
Y = sch.linkage(D, method='centroid')
Z = sch.dendrogram(Y, orientation='right')
axdendro.set_xticks([])
axdendro.set_yticks([])

# Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
index = Z['leaves']
R = R[index,:]
R = R[:,index]
im = axmatrix.matshow(R, aspect='auto', origin='lower',vmin=-1,vmax=+1)
axmatrix.set_xticks([])
axmatrix.set_yticks([])

# Plot colorbar.
axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
pylab.colorbar(im, cax=axcolor)

# Display and save figure.
fig.show()
fig.savefig('dendrogram.png')




#################


import matplotlib.pyplot as plt

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


############


import scipy
import scipy.cluster.hierarchy as sch
X = scipy.randn(100, 2)     # 100 2-dimensional observations
d = sch.distance.pdist(X)   # vector of (100 choose 2) pairwise distances
L = sch.linkage(d, method='complete')
ind = sch.fcluster(L, 0.5*d.max(), 'distance')


