# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:26:56 2015

@author: cgs567
"""
import matplotlib.pyplot as plt
from WormBox.ioworm import ls, pathWormPics
from WormBox.WormPlot import sidebysideplot
from mypystruct.utils import SaveLogger
#import cv2
import numpy as np
from skimage import io,img_as_float
folder=pathWormPics+'/Synthesized/CRFtest'
files=ls(folder,1,0)

#exp
I=np.zeros((10,15)).astype('float')
I[3:8,2:6]=0.5
I[3:8, 9:13]=1.0

#LOAD IMAGES
J=I.copy()
J[I==0.5]=1
J[I==1.0]=2
J=J.astype('int')

#Should make impossible with standard graph
I[3:8,2:6]=1.0
I[3:8, 9:13]=1.0


fx='./../Corral/Snyder/WormPics/Synthesized/CRFtest/2neurons.tif'
ftruth='./../Corral/Snyder/WormPics/Synthesized/CRFtest/truth_2neuron.tif'
img=img_as_float(io.imread(fx,0))
img/=img.max()
tru=io.imread(ftruth)
tru=tru.copy()-1#moves labels to starting at 0
#sidebysideplot([X,Xtru],horiz=False)



#from pystruct import learners
#from pystruct import models

from mypystruct import learners
from mypystruct import models
from mypystruct.models.template_crf import TemplateCRF

#SETUP

#bin_tru[bin_tru>1]=1#only 0's and 1's


n_states=3
#SIMPLE PYSTRUCT
X=img[np.newaxis,:,:,np.newaxis]
Y=tru[np.newaxis,:,:]

X=I[np.newaxis,:,:,np.newaxis]##For debug --artificial
Y=J[np.newaxis,:,:]

#gmm=np.histogram(I[J>0])

#setup template
dimx,dimy=X.shape[1:3]
v=np.ones((dimx,1))*np.linspace(-1,1,dimy)

v[v<0]=-1
v[v>0]=1

#inference_method='lp'
inference_method='ogm'
#inference_method='ad3'
#inference_method='qpbo'
crf=TemplateCRF(neighborhood=4,n_states=n_states,template=v,inference_method=inference_method)
#crf=models.GridCRF(neighborhood=4,n_states=n_states,inference_method=inference_method)
verbose=1
show_loss_every=1
save_every=1
experiment_name="template_CRF_2_neuron"
logger=SaveLogger(experiment_name + ".pickle", save_every=save_every)
tol=0.3
clf = learners.OneSlackSSVM(model=crf, C=10, inference_cache=2000,
                            tol=tol,show_loss_every=show_loss_every,
                            verbose=verbose,logger=logger)
clf.fit(X,Y)
print 'fitting done.. predicting now..'
Y_pred = np.array(clf.predict(X))
print 'Y_pred: '
print Y_pred
#sidebysideplot([img,bin_tru,Y_pred[0]],titles=['Image','Ground Truth','Prediction'],colormaps=['Greys','jet'])

Yc=[y[0].reshape(Y.shape) for y in clf.Yhat_log]#the Yc that create the constraints.
wc=[ww.round(2) for ww in clf.w_log]

print 'n_states = ', clf.model.n_states
print 'n_features = ',clf.model.n_features
print 'w is '
print clf.w
#print 'unaries are '
#print clf.w[:clf.model.n_states*clf.model.n_features]
#print 'pairwise params are '
#print clf.w[clf.model.n_states*clf.model.n_features:]

#try:
a=clf.model.joint_feature(X,Y)
b=clf.model.joint_feature(X,Y_pred)
b_loss=clf.model.loss(Y,Y_pred)
w=clf.w
constr=zip(*clf.constraints_)[0][1:]

Consistent=[(np.dot(w,djoint),clf.model.loss(Y,ybar)) for djoint,ybar in zip(constr,Yc)]

print '(w.T * djoint, loss)'
for c in Consistent:
    print c

u=clf.w[:clf.model.offset_pairwise]
p=clf.w[clf.model.offset_pairwise:clf.model.offset_template]
t=v.reshape(-1,1)*clf.w[clf.model.offset_template:]
#except:
#    pass