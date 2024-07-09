#include <stdio.h>
#include <iostream>
//#include <omp.h>
//#include <vector>
//#include <gsl/gsl_sf_legendre.h>

//#include <math.h>//for cos()
/*
#define PI 3.141592653

double sh(int m,int n,double theta,double phi){
	double x=gsl_sf_legendre_sphPlm(n,m,cos(phi));
	return x;
}

std::vector<double> compute_sphere(int m,int n, double theta,std::vector<double> phi){

std::vector<double> dst;
long int L=phi.size();
dst.resize(L);
//omp_set_num_threads(8);
//#pragma omp parallel for
for(long int i=0;i<L;i++){
	dst[i]=sh(m,n,theta,phi[i]);
}
return dst;

}
*/

main(void){

std::cout<<"hello world"<<std::endl;

/*
std::vector<double> phi;
for(int j=0;j<50;j++){
phi.push_back(j);
}

std::vector<double> dst=compute_sphere(1,2,22,phi);

for(long int d=0;d<dst.size();d++){
std::cout<<dst[d]<<std::endl;
}
*/
/*
double results[]={sh(0,0,0,0),sh(0,1,11.1,PI/4.0),sh(2,3,0,0.25)};

for(int i=0;i<3;i++){
std::cout<<results[i]<<std::endl;
}

std::cout<<"hello"<<std::endl;
*/
}
