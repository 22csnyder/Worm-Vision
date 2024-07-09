import numpy as np
cimport numpy as np


from libcpp.vector cimport vector


DTYPE=np.double#runtime type
ctypedef np.double_t DTYPE_t#compiletime type


cdef extern from "harm.h":
	cdef vector[DTYPE_t] compute_sphere(int,int,DTYPE_t,vector[DTYPE_t])
	cdef DTYPE_t sh(int,int,DTYPE_t,DTYPE_t)
	cdef void compute_sphere_by_reference(int,int,double*,double*,long,double*) 	
	cdef void dummy(double*)

def cy_Y(int m,int n,np.ndarray[DTYPE_t,ndim=3] Theta,np.ndarray[DTYPE_t,ndim=3] Phi):
	cdef np.ndarray[DTYPE_t,ndim=3] Dst=np.zeros_like(Theta)
	Len=Theta.size
	#dummy(&Dst[0,0,0])	
	compute_sphere_by_reference(<int> m,<int> n,&Theta[0,0,0],&Phi[0,0,0],<long> Len,&Dst[0,0,0])	
	return Dst

def cy_sph_har(int m,int n,double theta,double phi):
	return sh(m,n,theta,phi)


#Utility function to convert to vector from array
cdef vector[DTYPE_t] arrayToVector(np.ndarray[DTYPE_t,ndim=1] array):
	cdef long size =array.size
	cdef vector[DTYPE_t] vec
	cdef long i
	for i in range(size):
		vec.push_back(array[i])
	return vec



#cdef np.ndarray[DTYPE_t,ndim=1] vectorToArray(vector[DTYPE_t] vec):
def vectorToArray(vector[double] vec):
#	cdef np.ndarray[np.double_t,ndim=1] rr=np.zeros((vec.size(),),dtype=np.double)	
	#cdef <DTYPE_t*> p_data=vec
	cdef np.ndarray[DTYPE_t,ndim=1] arr=np.empty(vec.size(),dtype=DTYPE)
#For now I'm copying the information directly over. I could have the output array point to the same memory as vec, but what if vec goes out of scope?
#You could set up so that the memory is freed when the array is freed, but that needs to be done through a cdef class
	for i in range(vec.size()):
		arr[i]=vec[i]	
#arr.data=&vec[0]
	return arr


def py_compute_sphere(int m,int n,double theta, np.ndarray[DTYPE_t,ndim=1] phi):
	#j<double*> d_phi=phi.data
	#len=phi.size
	#vector[double] vec_phi=vector.assign(d_phi,d_phi+len)
	vec_phi=arrayToVector(phi)
	ans= compute_sphere(<int> m,<int> n,<double> theta, vec_phi) 
	np_ans=vectorToArray(ans)
	return np_ans




def my_compute(int m,int n,DTYPE_t theta, DTYPE_t phi):
	return sh(<int> m,<int> n,<double> theta,<double> phi)	

