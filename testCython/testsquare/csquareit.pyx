import numpy as np
cimport numpy as np


cdef extern from "squareit.h":
	cdef int bytwo(double* x,int len)
	cdef int parbytwo(double* x,int len)

def pysq(A):
	sum=0.0
	for x in A:
		sum+=x*x
	return sum

def npsq(A):
	return np.sum(A**2)

def csq(np.ndarray[np.double_t,ndim=1] A):
	return bytwo(<double*> A.data,<int> len(A))

def parcsq(np.ndarray[np.double_t,ndim=1] A):
	return parbytwo(<double*> A.data,<int> len(A))


