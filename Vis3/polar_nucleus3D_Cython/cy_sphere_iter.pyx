import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

DOUBLE=np.double#runtime
ctypedef np.double_t DOUBLE_t#equivalent compiletime type from cimport numpy

#I couldnt' find a np type that translates to C int
#INT=np.int
#ctypedef np.int_t INT_t

LONG=np.int
ctypedef np.int_t LONG_t

cdef extern from "iteration.h":
	cdef void do_iteration(
        DOUBLE_t* Radius,DOUBLE_t* I,
        LONG_t std_size,LONG_t ref_size,
        LONG_t* ref_shape,LONG_t* std_shape,
        LONG_t* ref_Dcrt_Angle,
        DOUBLE_t* ref_Distance,
        DOUBLE_t* Awt,LONG_t* Awt_shape,DOUBLE_t* Aml,DOUBLE_t* Wml,
        DOUBLE_t* Center,DOUBLE_t* c_est,DOUBLE_t r0,DOUBLE_t r_est,
        LONG_t* int_Center,LONG_t* ref_origin,
        DOUBLE_t u0,
        DOUBLE_t lamP,DOUBLE_t lamM,DOUBLE_t lamR,DOUBLE_t lamL,
        DOUBLE_t delta,DOUBLE_t tau,
        LONG_t n_threads,
        
	DOUBLE_t& u1,
        DOUBLE_t* delA,DOUBLE_t* delc,DOUBLE_t& delr0)


cdef extern from "getinfo.h":
	cdef void return_pixel_membership(
	DOUBLE_t* Radius,DOUBLE_t* I,
	LONG_t std_size, LONG_t ref_size,
	LONG_t* ref_shape,LONG_t* std_shape,
	LONG_t* ref_Dcrt_Angle,
	DOUBLE_t* ref_Distance,
	LONG_t* int_Center, LONG_t* ref_origin,
	LONG_t n_threads,
	LONG_t* bool_image)

def cy_get_seg(
	np.ndarray[DOUBLE_t,ndim=2] Radius,np.ndarray[DOUBLE_t,ndim=3] I,
	LONG_t std_size, LONG_t ref_size,
	np.ndarray[LONG_t,ndim=1] ref_shape,np.ndarray[LONG_t,ndim=1] std_shape,
	np.ndarray[LONG_t,ndim=3] ref_Dcrt_Angle,
	np.ndarray[DOUBLE_t,ndim=3] ref_Distance,
	np.ndarray[DOUBLE_t,ndim=1] Center,
	np.ndarray[LONG_t,ndim=1] ref_origin,
	LONG_t n_threads):#,
#	np.ndarray[LONG_t,ndim=3] bool_image):

	cdef np.ndarray[LONG_t,ndim=3] bool_image=np.zeros(ref_shape).astype(LONG)
	cdef np.ndarray[LONG_t,ndim=1] int_Center=np.round(Center).astype(LONG)

	return_pixel_membership(
		&Radius[0,0],&I[0,0,0],
		<LONG_t> std_size,<LONG_t> ref_size,
		&ref_shape[0],&std_shape[0],
		&ref_Dcrt_Angle[0,0,0],
		&ref_Distance[0,0,0],
		&int_Center[0],&ref_origin[0],
		<LONG_t> n_threads,
		&bool_image[0,0,0])
	return bool_image	

def cy_iteration(
	np.ndarray[DOUBLE_t,ndim=2] Radius, np.ndarray[DOUBLE_t,ndim=3] I,
	LONG_t std_size,LONG_t ref_size,
	np.ndarray[LONG_t,ndim=1] ref_shape,np.ndarray[LONG_t,ndim=1]std_shape,
	np.ndarray[LONG_t,ndim=3] ref_Dcrt_Angle,
	np.ndarray[DOUBLE_t,ndim=3] ref_Distance,
	np.ndarray[DOUBLE_t,ndim=2] Awt,np.ndarray[LONG_t,ndim=1] Awt_shape,
	np.ndarray[DOUBLE_t,ndim=1] Aml,np.ndarray[DOUBLE_t,ndim=1] Wml,
	np.ndarray[DOUBLE_t,ndim=1] Center, np.ndarray[DOUBLE_t,ndim=1] c_est,
	#np.ndarray[DOUBLE_t,ndim=1] int_Center,	
	DOUBLE_t r0,DOUBLE_t r_est,
	np.ndarray[LONG_t,ndim=1] ref_origin,
	DOUBLE_t lamP,DOUBLE_t lamM,DOUBLE_t lamR,DOUBLE_t lamL,
	DOUBLE_t delta,DOUBLE_t tau,
	LONG_t n_threads
	):


	cdef np.ndarray[DOUBLE_t,ndim=1] delA=np.zeros(Aml.shape[0],dtype=DOUBLE)
	cdef np.ndarray[DOUBLE_t,ndim=1] delc=np.zeros(3)
	cdef DOUBLE_t u1=0
	cdef DOUBLE_t delr0=0	


	cdef DOUBLE_t u0 =  0.05#Fixed to save comp time
	#I think I get lucky here is the only reason why this works
	cdef np.ndarray[LONG_t,ndim=1] int_Center=np.round(Center).astype(LONG)
	
	
	do_iteration(
		&Radius[0,0],&I[0,0,0],
		<LONG_t> std_size,<LONG_t> ref_size,
		&ref_shape[0],&std_shape[0],
		&ref_Dcrt_Angle[0,0,0],
		&ref_Distance[0,0,0],
		&Awt[0,0],&Awt_shape[0],&Aml[0],&Wml[0],
		&Center[0],&c_est[0],<double> r0,<double> r_est,
		&int_Center[0],&ref_origin[0],
		<DOUBLE_t> u0,
		<DOUBLE_t> lamP,<DOUBLE_t> lamM,<DOUBLE_t> lamR,<DOUBLE_t> lamL,
		<DOUBLE_t> delta,<DOUBLE_t> tau,
		<LONG_t> n_threads,
		u1,
		&delA[0],&delc[0],delr0
					)
	

	#print "u1 in cy_iteration",u1

	return u1,delA,delc,delr0


