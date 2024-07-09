# -*- coding: utf-8 -*-
"""
Created on Thu Sep 03 22:23:50 2015

@author: cgs567
"""

from WormBox.BaseClasses import FileReader
import numpy as np

#150109
#apo_file='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal/150109_CM_HighRes/150109_HighRes_ZIM294_L4_N40x_150902CS.apo'
#xmin=1174;xmax=1364+xmin#2538

#150311_20xW1
#apo_file='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/20x/W1/sudip piecewise annotations/150311_ZIM294_L4_W1-2_z60um_O20X_F630x150um_P11008x1000S178_resized_RAW_stack20150804SM.apo'
#xmin=2958;xmax=500+xmin#3458

#150311_40xW1
#apo_file='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W1/sudip piecewise annotations/150311_ZIM294_L4_W1-1_20150804SM-nonredundant.apo'
#xmin=2163;xmax=750+xmin#2813

#150311_20xW2
#apo_file='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/20x/W2/sudip piecewise annotations/150311_ZIM294_L4_W2-1_z60um_O20X_F630x150um_P11008x1000S174_resized_RAW_stack20150804SM.apo'
#xmin=2580;xmax=500+xmin#3080

#150311_40xW2
#apo_file='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150311_CM_20x_vs_40x_highRes/40x/W2/sudip piecewise annotations/150311_ZIM294_L4_W2-1_z42um_N40X_F350x100um_P11008x1000S174_resized_RAW_stack_SM20150803.apo'
#xmin=2364;xmax=750+xmin#3114

#150731
#apo_file='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015/150731_CM_Device_II_Stimulus/W1/c_150731_Glycerol_1M_W1_O20X_F630x150um_P11008x1000S210_m2_resize_stack_20150805SM.apo'
#xmin=;xmax=+xmin

#apo_file=
#xmin=;xmax=+xmin

#apo_file=
#xmin=;xmax=+xmin



Markers=FileReader(apo_file)
Markers.read_markers()

count=len([xx for xx in Markers.x if xmin<xx<xmax])
print 'n markers is ',count