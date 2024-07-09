#include <stdio.h>
#include "squareit.h"
#include <omp.h>


int parbytwo(double* x,int len){
int sum=0;
//tnum=omp_get_max_threads()
omp_set_num_threads(2);jjk
#pragma omp parallel for num_threads(2)
for(int i=0;i<len;i++){
int t=omp_get_thread_num();
//printf("thread number is %i\n",t);
sum+=x[i]*x[i];
}
return sum;
} 


int bytwo(double*x,int len){
int sum=0;
for(int i=0;i<len;i++){
sum+=x[i]*x[i];
}
return sum;
}

/*
int main ()
{
double x[19]={3,4,5,3,4,5,3,4,5,3,4,5,3,4,5,3,4,5,3};
parbytwo(x,19);
return 0;
}
*/



