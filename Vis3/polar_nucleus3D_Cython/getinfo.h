#ifndef GETINFO_H
#define GETINFO_H
#include <stdio.h>
#include <vector>
#include <tuple>

typedef std::tuple<int,int,int,int,long> point;


void return_pixel_membership(
        double* Radius,double* I,
        long std_size,long ref_size,
        long* ref_shape,long* std_shape,
        long* ref_Dcrt_Angle,
        double* ref_Distance,
        long* int_Center,long* ref_origin,
        long n_threads,
        long* bool_image
		);  


#endif
