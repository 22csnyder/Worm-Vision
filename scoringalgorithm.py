# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 19:09:51 2015

@author: melocaladmin
"""
import matplotlib.pyplot as plt
import numpy as np
import math

def score(K):
    m,n=K.shape
    _K=K.copy()
    def f(x):
        loc=np.linspace(0,len(x)-1,len(x))
        mew=loc.dot(x)/np.sum(x)
        var=((loc-mew)**2).dot(x)/np.sum(x)
        return math.sqrt(var)
    for i in range(m):
        _K[i,:]/=f(_K[i,:])
    for j in range(n):
        _K[:,j]/=f(_K[:,j])
    means=[[np.mean(k) for k in _K],[np.mean(k) for k in _K.transpose()]]
    for i in range(m):
        for j in range(n):
            _K[i,j]/=math.sqrt(means[0][i]*means[1][j])
#    for i in range(m):
#        _K[i,:]/=np.mean(_K[i,:])
#    for j in range(n):
#        _K[:,j]/=np.mean(_K[:,j])    
    
    return _K
    
M=np.random.rand(6,6)
M=M/M.mean()
plt.imshow(M,vmin=M.min(),vmax=M.max(),interpolation = 'none')
plt.title('M')
plt.draw()
plt.figure()
N=score(M)
plt.imshow(N,vmin=N.min(),vmax=N.max(),interpolation = 'none')
plt.title('N')

#N=score(N);plt.imshow(N,vmin=N.min(),vmax=N.max(),interpolation = 'none')
