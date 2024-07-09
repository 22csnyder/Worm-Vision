#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <vector>
#include <math.h>

#include "getinfo.h"

void return_pixel_membership(double* Radius,double* I,
        long std_size,long ref_size,
        long* ref_shape,long* std_shape,
        long* ref_Dcrt_Angle,
        double* ref_Distance,
        long* int_Center,long* ref_origin,
        long n_threads,
        //Return Values
        long* bool_image){//assume bool image starts out as zeros
omp_set_num_threads(n_threads);
size_t* prefix;
double interior_intensity=0;
long interior_count=0;
	int ithread=omp_get_thread_num();
	int nthreads=omp_get_num_threads();
	std::vector<point> bdry_private;
	#pragma omp for schedule(static) nowait collapse(3)
	//#pragma omp for schedule(dynamic) nowait collapse(3)//seems slower 10x than static
	for(int i=0;i<ref_shape[0];i++){
	for(int j=0;j<ref_shape[1];j++){
	for(int k=0;k<ref_shape[2];k++){
		int i2=i+int_Center[0]-ref_origin[0];
		int j2=j+int_Center[1]-ref_origin[1];
		int k2=k+int_Center[2]-ref_origin[2];
		if(i2>0 && i2<std_shape[0] &&
	       	   j2>0 && j2<std_shape[1] &&
		   k2>0 && k2<std_shape[2])
		{
			long ref_flat=k+j*ref_shape[2] + i*ref_shape[2]*ref_shape[1];		
			long std_flat=k2+j2*std_shape[2]+i2*std_shape[2]*std_shape[1];
			int ang=ref_Dcrt_Angle[ref_flat];
			double R=Radius[ang];
			double dist=ref_Distance[ref_flat];
			
			if(dist<R){
				interior_count++;
				interior_intensity+=I[std_flat];
				bool_image[std_flat]=1;//set to 1 if inside
			}
		}
	}}}
	/*#pragma omp critical
	{ 
	std::cout<<"thread "<<ithread<<"bdrysize "<<bdry_private.size()<<std::endl;
	}*/
//std::cout<<"find_memb: intensity "<<interior_intensity<<" count "<<interior_count<<std::endl;
//std::cout<<"total bdry size "<<bdry_idx.size()<<std::endl;
}





