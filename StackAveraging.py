# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:29:27 2015

@author: cgs567
"""


from WormBox.BaseClasses import Worm, Worm4D , Environment
import numpy as np

read_dir='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal-2015'
write_dir='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/TimeAveraging/150605 Not Anesthetized'
env=Environment(read_dir,write_dir)



folder='/150605_1824_Vol_Img_No_Stimulus_W5/stacks'
env.read_dir+=folder

worm=Worm(env)

wormList=[]
for i in range(163,183):
    worm=Worm(env)
    worm.read('worm'+str(i)+'.tif')
    wormList.append(worm)
    
tstack=np.vstack( [w.Stack[np.newaxis,:] for w in wormList])


dstWorm=Worm(env)
tstack=tstack.astype(np.float)

S=tstack.sum(0)
dstStack=tstack.sum(0)/len(tstack)#Hope no overflow
dstWorm.Stack=dstStack.astype(np.uint16)
dstWorm.save('150605_1824_Vol_Img_No_Stimulus_W5 vols163-183')