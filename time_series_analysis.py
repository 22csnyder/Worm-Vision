# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 08:50:54 2015

@author: csnyder
"""

from PlotSegmentParameters import get_segment_list


from WormBox.WormPlot import getsquarebysideaxes

#
#readdir='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/seg_patches'
#marker=0
#
#ps='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/Seg1/Segment_0.pkl'
#
#pf='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/Seg1/'
#
#S=get_segment_list(pf)
#
#
#s=S[0]
#
#path='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/Seg1/S.pkl'


path='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/150902_1703_Vol_Img_Dev_II_No_Stimulus_W1/Seg1/segments.pkl'
import dill
with open(path,'r') as handle:
    segments=dill.load(handle)
    
s=segments


key=s.keys()[0]

for key in s.keys():
    neuron=s[key]
    if 'notes' in neuron.keys():
        print key,neuron['notes']



import matplotlib.pyplot as plt



fig,axes=getsquarebysideaxes(len(s))
for key,ax in zip(s.keys(),axes):
    intensity=s[key]['intensity'].values()
    time_points=s[key]['intensity'].keys()
    t,y=zip(*sorted(zip(time_points,intensity),key=lambda x:int(x[0])))
#    ax.plot(t,y,'b',t,y,'ko',mfc='none')
    ax.plot(t,y,'b',t,y)
    ax.set_title(key)
    ax.set_xlim(0,232)#232 timepoints
    ax.set_ylim(0,6000)

plt.show()
#plt.plot(t,y)







