# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 07:30:26 2015

@author: christopher


Warning this is a python2 file
"""

from WormBox.ioworm import ls
from skimage import io
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import opengm

picdir='./../TestPics'
pics=ls(picdir)

img=io.imread(pics[1],'asgrey')
io.imshow(img)
#io.show()


#shape=img.shape
#numLabels=3
#unaries=np.random.rand(shape[0],shape[1],numLabels)
#potts=opengm.PottsFunction([numLabels,numLabels],0.0,0.5)
#gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)
#opengm.visualizeGm( gm,plotFunctions=False,plotNonShared=True,relNodeSize=0.9)

#print "number of factors:",gm.numberOfFactors
#print "number of factors of node 0: ",gm.numberOfFactorsOfVariable(0)
#print "number of factors of node 4: ",gm.numberOfFactorsOfVariable(4)

shape=img.shape
dimx,dimy=shape[0],shape[1]
numVar=dimx*dimy
numLabels=3
beta=0.1
numberOfStates=np.ones(numVar,dtype=opengm.index_type)*numLabels
gm=opengm.graphicalModel(numberOfStates)

#add Unary factors, assign Potts function to the factors
for y in range(dimy):
    for x in range(dimx):
        f=np.ones(2,dtype=np.float32)
        f[0]=img[x,y]
        f[1]=1.0-img[x,y]
        fid=gm.addFunction(f)
        gm.addFactor(fid,(x*dimy+y,))
        
#Adding binary function and factors
#create the pairwise function (Potts function)
f=np.ones(pow(numLabels,2),dtype=np.float32).reshape(numLabels,numLabels)*beta
for l in range(numLabels):
    f[l,l]=0
fid=gm.addFunction(f)
#create pairwise factors for the whole grid, and
#assign the Potts function created above, to each new factor.
for y in range(dimy):
    for x in range(dimx):
#add a factor between each pair of neighboring nodes.
        if(x+1<dimx):
#add a factor between a node and its neighbor on the right
            gm.addFactor(fid,np.array([x*dimy+y,(x+1)*dimy+y],dtype=opengm.index_type))
        if(y+1<dimy):
#add a factor between a node and its neighbor above.
            gm.addFactor(fid,[x*dimy+y,x*dimy+(y+1)])


imgplot=[]
class PyCallback(object):
    def appendLabelVector(self,labelVector):
        #save the labels at each iteration, to examine later.
        labelVector=labelVector.reshape(self.shape)
        imgplot.append([labelVector])
    def __init__(self,shape,numLabels):
        self.shape=shape
        self.numLabels=numLabels
        matplotlib.interactive(True)
    def checkEnergy(self,inference):
        gm=inference.gm()
        #the arg method returns the (class) labeling at each pixel.
        labelVector=inference.arg()
        #evaluate the energy of the graph given the current labeling.
        print "energy ",gm.evaluate(labelVector)
        self.appendLabelVector(labelVector)
    def begin(self,inference):
        print "beginning of inference"
        self.checkEnergy(inference)
    def end(self,inference):
        print "end of inference"
    def visit(self,inference):
        self.checkEnergy(inference)

inf=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(steps=6,damping=0.3))
#parameter=opengm.InfParam(damping=0.1)
callback=PyCallback(shape,numLabels)
visitor=inf.pythonVisitor(callback,visitNth=1)
inf.infer(visitor)


fig = plt.figure(figsize=(16, 12))
for (counter, im) in enumerate(imgplot[0:6]):
    a=fig.add_subplot(3,2,counter+1)
    plt.imshow(im[0],cmap=matplotlib.cm.gray, interpolation="nearest")
    plt.draw()







