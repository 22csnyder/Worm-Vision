import numpy as np

from timeit import Timer

from sphhar import py_compute_sphere as pcs
from sphhar import my_compute as mc





def f():
	for a in A:
		out=mc(m,n,theta,a)


#if __name__=='__main__':

m=1
n=3
theta=.45
A=np.array(range(150000)).astype('double')


def par():
    pcs(m,n,theta,A)

#omp=Timer('pcs(m,n,theta,A)',setup='A=A;m=m;n=n;theta=theta')
omp=Timer('par()',setup='from __main__ import par')

single=Timer('f()',setup='from __main__ import f')
print 'single threaded time'
print single.timeit(100)
print 'multithread'
print omp.timeit(100)



