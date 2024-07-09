# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:20:15 2015

@author: csnyder
"""

import numpy as np
import theano
import theano.tensor as T

def floatX(X):
    return np.asarray(X,dtype=theano.config.floatX)
    
def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape)*0.01))
    
a=T.dscalar('a')
b=T.dscalar('b') 
c=a+b
f=theano.function([a,b],c)

e=T.dscalar('b')
e.tag.test_value=3
d=f(e,e)
    
    
#h=T.scalar('h')
#focii=h
##focii_coordinates = theano.function([ h, Pa(offsets,default=np_gap) ],focii , allow_input_downcast=True)
#focii_coordinates = theano.function([ h ],focii)
#theano.config.compute_test_value='warn'
#h_1=T.scalar('h_1')
##h_1.tag.test_value=floatX(np.array([0.4,0.6]))
#h_1.tag.test_value=floatX(7)
#a=focii_coordinates(h_1)
    
    
    
#A = T.matrix('A')
#x = T.vector('x')
#b = T.vector('b')
#y = T.dot(A, x) + b
## Note that squaring a matrix is element-wise
#z = T.sum(A**2)
## theano.function can compute multiple things at a time
## You can also set default parameter values
## We'll cover theano.config.floatX later
#b_default = np.array([0, 0], dtype=theano.config.floatX)
#linear_mix = theano.function([A, x, theano.Param(b, default=b_default)], [y, z]) 
    




#if you don't put ('a') it counts as a "function" for some reason
#also it won't let you access tag.test_value    
#a=T.dscalar('a')
##don't set a,b explicitly: error: #a=3.0#b=5.0
#b=T.dscalar('b') 
#c=a+b
#e=T.dscalar('e')
#f=c+e
#f=theano.function([a,b],c)






#foo=T.scalar('foo')
#def square(x):
#    return x**2
#bar = square(foo)
##print bar.eval({foo: 3})
#f=theano.function([foo],bar)


#shared_var = theano.shared(np.array([[1, 2], [3, 4]], dtype=theano.config.floatX))
#shared_squared = shared_var**2
#print bar.eval({foo:3})




######Interpolate
#####Using scipy won't work here for now, but there are options for wrapping such a function
#from scipy.interpolate import Rbf
#x,y=np.indices(I.shape)+0.5
#rbf=Rbf(x.flatten(),y.flatten(),I.flatten(),function='linear')#interpolate
######Visualize interpolation
#xi,yi=np.mgrid[0:4:50j,0:4:50j]
#rbi=rbf(xi.flatten(),yi.flatten())
#R=rbi.reshape(50,50)

#bvec3=T.TensorType(dtype=theano.config.floatX,broadcastable=(True,False))
#broadvector3=T.TensorType(dtype=theano.config.floatX,broadcastable=(True,True,False) )    
#u=broadvector3.filter(floatX(np.array([1.5,1.0]).reshape(1,2)))
