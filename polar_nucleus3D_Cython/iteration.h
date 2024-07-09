#ifndef ITERATION_H
#define ITERATION_H
#include <stdio.h>
#include <vector>
#include <tuple>

typedef std::tuple<int,int,int,int,long> point;
//typedef std::tuple<long,long,long,long,long> point;
/*
void find_pixel_membership(double* Radius,double* I,
	long std_size,long ref_size,
	int* ref_shape,int* std_shape,
	int* ref_Dcrt_Angle,
	double* ref_Distance,
	double delta,
	int* int_Center,int* ref_origin,
	int n_threads,
	double u0,
	//Return Values
	std::vector<point>& bdry_idx,
	double& u1);

	

void calculate_changes(
	//inputes
	const std::vector<point>& bdry_idx,
	double* I,double* Awt,int* Awt_shape,double*Aml,double*Wml,
	double* Center,double* c_est,double r0,double r_est,
	double u0,double u1,
	double lamP,double lamM,double lamR,double lamL,
	//Return Values
	double* delA,double* delc,double& delr0);
*/
void do_iteration(
        double* spacing,double* Radius,double* I,
        long std_size,long ref_size,
        long* ref_shape,long* std_shape,
        long* ref_Dcrt_Angle,
        double* ref_Distance,
        double* Awt,long* Awt_shape,double* Aml,double* Wml,
        //double* Awt,long* Awt_shape,double* ERS,double* Aml,double* Wml,
        double* Center,double* c_est,double r0,double r_est,
        long* int_Center,long* ref_origin,
        double u0,
        double lamP,double lamM,double lamR,double lamL,
        double delta,double tau,
        long n_threads,
        
	double& u1,
        double* delA,double* delc,double& delr0
        		);


#endif
