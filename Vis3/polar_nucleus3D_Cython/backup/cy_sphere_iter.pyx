import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

DOUBLE=np.double#runtime
ctypedef np.double_t DOUBLE_t#equivalent compiletime type from cimport numpy

INT=np.int
ctypedef np.int_t INT_t

LONG=np.int
ctypedef np.int_t LONG_t

cdef extern from "iteration.h":
	cdef void do_iteration(
        DOUBLE_t* Radius,DOUBLE_t* I,
        LONG_t std_size,LONG_t ref_size,
        INT_t* ref_shape,INT_t* std_shape,
        INT_t* ref_Dcrt_Angle,
        DOUBLE_t* ref_Distance,
        DOUBLE_t* Awt,INT_t* Awt_shape,DOUBLE_t* Aml,DOUBLE_t* Wml,
        DOUBLE_t* Center,DOUBLE_t* c_est,DOUBLE_t r0,DOUBLE_t r_est,
        INT_t* int_Center,INT_t* ref_origin,
        DOUBLE_t u0,
        DOUBLE_t lamP,DOUBLE_t lamM,DOUBLE_t lamR,DOUBLE_t lamL,
        DOUBLE_t delta,DOUBLE_t tau,
        INT_t n_threads,
        
	DOUBLE_t& u1,
        DOUBLE_t* delA,DOUBLE_t* delc,DOUBLE_t& delr0
				)

def cy_iteration(
	np.ndarray[DOUBLE_t,ndim=2] Radius, np.ndarray[DOUBLE_t,ndim=3] I,
	LONG_t std_size,LONG_t ref_size,
	np.ndarray[INT_t,ndim=1] ref_shape,np.ndarray[INT_t,ndim=1]std_shape,
	np.ndarray[INT_t,ndim=3] ref_Dcrt_Angle,
	np.ndarray[DOUBLE_t,ndim=3] ref_Distance,
	np.ndarray[DOUBLE_t,ndim=2] Awt,np.ndarray[INT_t,ndim=1] Awt_shape,
	np.ndarray[DOUBLE_t,ndim=1] Aml,np.ndarray[DOUBLE_t,ndim=1] Wml,
	np.ndarray[DOUBLE_t,ndim=1] Center, np.ndarray[DOUBLE_t,ndim=1] c_est,
	DOUBLE_t r0,DOUBLE_t r_est,
	np.ndarray[INT_t,ndim=1] ref_origin,
	DOUBLE_t lamP,DOUBLE_t lamM,DOUBLE_t lamR,DOUBLE_t lamL,
	DOUBLE_t delta,DOUBLE_t tau,
	INT_t n_threads,

	DOUBLE_t u1,
	np.ndarray[DOUBLE_t,ndim=1] delA,
	np.ndarray[DOUBLE_t,ndim=1] delc,
	DOUBLE_t delr0):


	u0=0.05#Fixed to save comp time
	cdef np.ndarray[INT_t,ndim=1] int_Center=np.round(Center).astype(INT)

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
		<INT_t> n_threads,
		u1,
		&delA[0],&delc[0],delr0
					)

