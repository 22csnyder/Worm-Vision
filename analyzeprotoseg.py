# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 10:03:19 2015

This file is to investigate segmentations carried out by protoseg.py file. 
Hopefully it can give feedback as to what works and what doesn't work
Feb25,2015
@author: cgs567
"""

from protoseg import Registrar
import matplotlib.pyplot as plt





#path='./../WormPics/Synthesized/MegaplusTestData'
path='./../WormPics/Synthesized/2DtestData'

regis=Registrar(path)
fig,axes=regis.show_segmentation()
R=regis.Align.findBestAlignment()
print(R)



#def Kplot(setofK):
#    fig,axes=plt.subplots(2,3)
#    axes=axes.flatten()
#    titles=['0','1','2','3','4','5']
#    fig.subplots_adjust(hspace=.001,wspace=0.001)
#    fig.tight_layout(pad=0,w_pad=0.0001,h_pad=0.0001)
#
#    for p,a,t in zip(setofK,axes,titles):
#        a.imshow(p,vmin=p.min(),vmax=p.max())
#        a.set_title(t)
#    fig.canvas.draw()


