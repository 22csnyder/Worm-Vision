# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 21:16:06 2015

@author: christopher
"""

import numpy as np
from WormBox.ioworm import ls,pathWormPics
from WormBox.WormPlot import sidebysideplot
#import cv2
from skimage import io,img_as_float
import matplotlib.pyplot as plt

from pystruct import learners
from pystruct import models


#SETUP
synth=pathWormPics+'/Synthesized/CRFtest'
files=['/2neurons.tif','/truth_2neuron.tif']

img=img_as_float(io.imread(synth+files[0]))
img/=img.max()
tru=io.imread(synth+files[1])

bin_tru=tru.copy()-1#moves labels to starting at 0
bin_tru[bin_tru>1]=1#only 0's and 1's


#SIMPLE PYSTRUCT
X=img[np.newaxis,:,:,np.newaxis]
Y=bin_tru[np.newaxis,:,:]



crf=models.GridCRF(neighborhood=4)
clf = learners.OneSlackSSVM(model=crf, C=10, inference_cache=100,tol=.1)
clf.fit(X,Y)

Y_pred = np.array(clf.predict(X))

sidebysideplot([img,bin_tru,Y_pred[0]],titles=['Image','Ground Truth','Prediction'],colormaps=['Greys','jet'])

#setup unique feature:::
n_labels=3
y=tru

where=[np.array(map(np.mean,np.where(y==l))) for l in range(1,n_labels)]
vector=where[0]-where[1]
nvec=vector/(np.sum(vector**2))**(0.5)


def example_spatial_feature(x,y):#esf
    where=[np.array(map(np.mean,np.where(y==l))) for l in range(1,n_labels)]
    vector=where[0]-where[1]
    return vector
    
esf_length=2

from spatial_mcrf import CustomJointFeature,SpatialmCRF

myfea=CustomJointFeature(example_spatial_feature,esf_length)



#MSC useful
#from pystruct.utils import make_grid_edges

#crosses example
#from pystruct.datasets import generate_crosses_explicit
#from pystruct.utils import expand_sym
#X, Y = generate_crosses_explicit(n_samples=50, noise=10)


#plot directional grid example code
#from pystruct.models import DirectionalGridCRF
#import pystruct.learners as ssvm
#from pystruct.datasets import generate_blocks_multinomial
#X, Y = generate_blocks_multinomial(noise=2, n_samples=20, seed=1)

#crf = DirectionalGridCRF(inference_method="qpbo", neighborhood=4)
#clf = ssvm.OneSlackSSVM(model=crf, n_jobs=-1, inference_cache=100, tol=.1)
#clf.fit(X, Y)

##Digits example
#from sklearn.cross_validation import train_test_split
#from sklearn.datasets import load_digits
#digits = load_digits()
#X, y_org = digits.data, digits.target
#X /= X.max()
#
## Make binary task by doing odd vs even numers.
#y = y_org % 2
#
## Make each example into a tuple of a single feature vector and an empty edge
## list
#X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
#Y = y.reshape(-1, 1)
#
#X_train_, X_test_, X_train, X_test, y_train, y_test, y_org_train, y_org_test =train_test_split(X_, X, Y, y_org, test_size=.5)

# First, perform the equivalent of the usual SVM.  This is represented as
# a CRF problem with no edges.

#pbl = GraphCRF(inference_method='unary')
## We use batch_size=-1 as a binary problem can be solved in one go.
#svm = NSlackSSVM(pbl, C=1, batch_size=-1)
#
#svm.fit(X_train_, y_train)
