#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <vector>
#include <math.h>
#include <tuple>

typedef std::tuple<int,int,int> i3tuple;

std::vector<int> do_parallel(){
omp_set_num_threads(8);
std::vector<int> vec;
size_t *prefix;
#pragma omp parallel
{
    int ithread  = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    #pragma omp single
    {
        prefix = new size_t[nthreads+1];
        prefix[0] = 0;
    }
    std::vector<int> vec_private;
    #pragma omp for schedule(static) nowait
    for(int i=0; i<100; i++) {
        vec_private.push_back(i+100*ithread);
    }
    prefix[ithread+1] = vec_private.size();
    #pragma omp barrier
    #pragma omp single 
    {
        for(int i=1; i<(nthreads+1); i++) prefix[i] += prefix[i-1];
        vec.resize(vec.size() + prefix[nthreads]);
    }
    std::copy(vec_private.begin(), vec_private.end(), vec.begin() + prefix[ithread]);
}
delete[] prefix;

return vec;
}

int main(void){

int x;int y;int z;
i3tuple t (2,5,7);
std::tie(x,y,z) = t;

printf("%d",x);
printf("%d",y);
printf("%d\n",z);


/*
std::vector<int> v = do_parallel();
for(int i=0;i<v.size();i++){
std::cout<<v[i]<<std::endl;
}
*/ 

/*
omp_set_num_threads(16)
#pragma omp parallel reduction(merge:v)
for(int i=0;i<10000;i++){
std::vector<int> v[
*/

//std::cout<<sizeof(int)<<std::endl;

}

