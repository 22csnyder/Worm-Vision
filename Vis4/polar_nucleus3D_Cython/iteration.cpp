#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <vector>
#include <math.h>
#include "iteration.h"

#define PI 3.141592653

void find_pixel_membership(double* Radius,double* I,
        long std_size,long ref_size,
        long* ref_shape,long* std_shape,
        long* ref_Dcrt_Angle,
        double* ref_Distance,
        double delta,
        long* int_Center,long* ref_origin,
        long n_threads,
        double u0,
        //Return Values
        std::vector<point>& bdry_idx,
        double& u1){
omp_set_num_threads(n_threads);
size_t* prefix;
double interior_intensity=0;
long interior_count=0;
#pragma omp parallel reduction(+:interior_intensity,interior_count)
{
	int ithread=omp_get_thread_num();
	int nthreads=omp_get_num_threads();
	#pragma omp single
	{
		prefix=new size_t[nthreads+1];
		prefix[0]=0;
	}
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
			
/*if(i==20&&j==35&&k==40){
std::cout<<"i j k "<<i<<j<<k<<std::endl;
std::cout<<"i2 j2 k2 "<<i2<<j2<<k2<<std::endl;
std::cout<<"std_flat "<<std_flat<<" ref_flat "<<ref_flat<<std::endl;
}*/
			if(dist<R){
				interior_count++;
				interior_intensity+=I[std_flat];
			}
			if(dist<=R+delta && dist>=R-delta){
				bdry_private.push_back(point(i2,j2,k2,ang,std_flat));
			}
		}
	}}}
	prefix[ithread+1]=bdry_private.size();
	#pragma omp barrier
	#pragma omp single
	{
		for(int i=1;i<(nthreads+1);i++) prefix[i]+=prefix[i-1];
		bdry_idx.resize(prefix[nthreads]);
	}
	std::copy(bdry_private.begin(),bdry_private.end(),bdry_idx.begin()+prefix[ithread]);
	/*#pragma omp critical
	{ 
	std::cout<<"thread "<<ithread<<"bdrysize "<<bdry_private.size()<<std::endl;
	}*/
}
delete[] prefix;
u1=interior_intensity/interior_count;
//std::cout<<"find_memb: intensity "<<interior_intensity<<" count "<<interior_count<<std::endl;
//std::cout<<"total bdry size "<<bdry_idx.size()<<std::endl;
}

void calc_forces(std::vector<double>& distributedF,
	const std::vector<point>& bdry_idx,
	double* I, double u1, double u0,double lamP){
long std_flat;
for(long b=0;b<bdry_idx.size();b++){
	std::tie(std::ignore,std::ignore,std::ignore,std::ignore,std_flat)=bdry_idx[b];
	double val=I[std_flat];
	double f=pow(val-u1,2)-pow(val-u0,2);
	distributedF[b]=lamP*f;	
}}

void calc_delA(const std::vector<double>& distributedF,double* Radius,double r0,
		const std::vector<point>& bdry_idx,
		double* Awt,long* Awt_shape,double* ERS,double* Aml,
		double*Wml,double lamL,
		double*delA,double* spacing){
/*//Some stuff I tried with the spacing didn't really work:
int s;//determines where smoothening starts

//if (spacing[0]==spacing[1] && spacing[1]==spacing[2]){
if(true){//debug
std::cout<<"spacing is uniform\n";
s=0;
}
else {
s=3;
int ix=0;int iy=2;int iz=1;//(-1,1),(0,1),(1,1)
double sigx=spacing[2];double sigy=spacing[1];double sigz=spacing[0];//z,y,x order
double wx=Wml[ix];double wy=Wml[iy];double wz=Wml[iz];
double ax=Aml[ix];double ay=Aml[iy];double az=Aml[iz];
double b=sqrt( 3.0/(4.0*PI) );

double xyskew=( (r0+b*ax)/sigx - (r0+b*ay)/sigy );
double xzskew=( (r0+b*ax)/sigx - (r0+b*az)/sigz );
double yzskew=( (r0+b*ay)/sigy - (r0+b*az)/sigz );
double delx=2*b*wx/sigx*(xyskew+xzskew);
double dely=2*b*wy/sigy*(yzskew-xyskew);
double delz=2*b*wz/sigz*(-1)*(xzskew+yzskew); 

*/
/*double xyskew=( sigy*(r0+b*ax) - sigx*(r0+b*ay) );double yxskew=-xyskew;
double xzskew=( sigz*(r0+b*ax) - sigx*(r0+b*az) );double zxskew=-xzskew;
double yzskew=( sigz*(r0+b*ay) - sigy*(r0+b*az) );double zyskew=-yzskew;
double delx=2*b*wx*sigy*xyskew + 2*b*wx*sigz*xzskew;
double dely=2*b*wy*sigx*yxskew + 2*b*wy*sigz*yzskew;
double delz=2*b*wz*sigx*zxskew + 2*b*wz*sigy*zyskew;

delA[ix]+=delx;delA[iy]+=dely;delA[iz]+=delz;
}
*/


long nF=Awt_shape[0];
long n_angles=Awt_shape[1];

for(long freq=0;freq<nF;freq++){
//if(freq>=s){delA[freq]+=2.0*Aml[freq]*Wml[freq]*lamL;}//don't penalize eccentricity
delA[freq]+=2.0*Aml[freq]*Wml[freq]*lamL;
long prefix=freq*n_angles;
for(int b=0;b<bdry_idx.size();b++){
int angle;
std::tie(std::ignore,std::ignore,std::ignore,angle,std::ignore)=bdry_idx[b];
//delA[freq]+= ( distributedF[b] -  lamL*2*Wml[freq]*(r0*ERS[angle] - Radius[angle]) )  * Awt[prefix+angle] ;
delA[freq]+= ( distributedF[b])  * Awt[prefix+angle] ;
}} 
} 

void calc_delc(const std::vector<double>& distributedF,
		const std::vector<point>& bdry_idx,
		double* Center,double* c_est, double lamM,
		double* delc){
int i;int j;int k;
for(long b=0;b<bdry_idx.size();b++){
	double force=distributedF[b];
	std::tie(i,j,k,std::ignore,std::ignore)=bdry_idx[b];
	double di=(double) i - Center[0];
	double dj=(double) j - Center[1];
	double dk=(double) k - Center[2];
	double norm=sqrt(pow(di,2)+pow(dj,2)+pow(dk,2));
	di/=norm;
	dj/=norm;
	dk/=norm;
	delc[0]+=force*di;
	delc[1]+=force*dj;
	delc[2]+=force*dk;
}
delc[0]+=lamM*2.0*(Center[0]-c_est[0]);
delc[1]+=lamM*2.0*(Center[1]-c_est[1]);
delc[2]+=lamM*2.0*(Center[2]-c_est[2]);	
}

void calc_delr0(const std::vector<double>& distributedF,
	double r0,double r_est,double lamR,
	double& delr0){
double radial_force=0;
for(long b=0;b<distributedF.size();b++){
radial_force+=distributedF[b];
}
delr0=0;
delr0+=radial_force + lamR*2.0*(r0-r_est);
}

void calculate_changes(
        const std::vector<point>& bdry_idx,double* Radius,
        double* I,double* Awt,long* Awt_shape,double* ERS,double* Aml,double*Wml,
        double* Center,double* c_est,double r0,double r_est,
        double u0,double u1,
        double lamP,double lamM,double lamR,double lamL,
        double* delA,double* delc,double& delr0,double* spacing){

std::vector<double> distributedF (bdry_idx.size(),0.0);
calc_forces(distributedF,bdry_idx,I,u1,u0,lamP);
calc_delA(distributedF,Radius,r0,bdry_idx,Awt,Awt_shape,ERS,Aml,Wml,lamL,delA,spacing);
calc_delc(distributedF,bdry_idx,Center,c_est,lamM,delc);
calc_delr0(distributedF,r0,r_est,lamR,delr0);
}

void do_iteration(
        double* spacing,double* Radius,double* I,
        long std_size,long ref_size,
        long* ref_shape,long* std_shape,
        long* ref_Dcrt_Angle,
        double* ref_Distance,
        double* Awt,long* Awt_shape,double* ERS,double* Aml,double* Wml,
        double* Center,double* c_est,double r0,double r_est,
        long* int_Center,long* ref_origin,
        double u0,
        double lamP,double lamM,double lamR,double lamL,
        double delta,double tau,
        long n_threads,
        //Return Values
        double& u1,
        double* delA,double* delc,double& delr0){



//The whole point of this wrapper function is that I didn't want to have to deal with
//the python code seeing std::vector<std::tuple> 
std::vector<point> bdry_idx;
find_pixel_membership(Radius,I,std_size,ref_size,ref_shape,std_shape,
	ref_Dcrt_Angle,ref_Distance,delta,int_Center,ref_origin,n_threads,u0,bdry_idx,u1);
calculate_changes(bdry_idx,Radius,I,Awt,Awt_shape,ERS,Aml,Wml,
	Center,c_est,r0,r_est,u0,u1,
	lamP,lamM,lamR,lamL,
	delA,delc,delr0,spacing);
}


int main(){

return 0;
}






