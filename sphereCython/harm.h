#include <vector>

double sh(int m,int n, double theta, double phi);

std::vector<double> compute_sphere(int m,int n,double theta, std::vector<double> phi);

void compute_sphere_by_reference(int m,int n,double* theta,double* phi,long Len,double* Y);

void dummy(double* x);
