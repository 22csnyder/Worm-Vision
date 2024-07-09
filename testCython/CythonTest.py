# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:12:08 2015

@author: christopher
"""
import extra.square.testquare.csquareit as s
#import testCython.square.testsquare.csquareit as s
#import testCython.csquareit as s
import numpy as np
import timeit

n=10000000
A=np.linspace(1,n,n)

niter=1
t_csq=timeit.Timer('s.csq(A)',setup='from __main__ import s,A').timeit(niter)
t_np=timeit.Timer('np.sum(A**2)',setup='from __main__ import np,A').timeit(niter)
t_np2=timeit.Timer('s.npsq(A)',setup='from __main__ import s,A').timeit(niter)
t_py=timeit.Timer('s.pysq(A)',setup='from __main__ import s,A').timeit(niter)
t_parcsq=timeit.Timer('s.parcsq(A)',setup='from __main__ import s,A').timeit(niter)

output=[('pure python',t_py),('pure numpy',t_np),('numpy from pyx',t_np2),
        ('sequential c++',t_csq),('2threaded c++',t_parcsq)]

for o in output: print(o)
        
