#include "dim.cuh"
#include "dim.hxx"
#include "timer.hxx"
#include "cuda_check.cuh"

__constant__ int d_n[3];
__constant__ double d_xmin[3];
__constant__ double d_dx[3];
__constant__ double d_lambda[3];

__global__
void symbol()
{
  printf("symbol : dx = %f %f %f\n", d_dx[0], d_dx[1], d_dx[2]);
  printf("symbol : n  = %d %d %d\n", d_n[0], d_n[1], d_n[2]);
}

void setDims(const int *h_n, 
             const double *h_xmin, 
             const double *h_dx, 
             const double *h_lambda)
{
    Timer & T = GetTimer(5); T.start();
    CUDA_CHECK_OP(cudaMemcpyToSymbol(d_n, h_n, 3 * sizeof(int)));
    CUDA_CHECK_OP(cudaMemcpyToSymbol(d_xmin, h_xmin, 3 * sizeof(double)));
    CUDA_CHECK_OP(cudaMemcpyToSymbol(d_dx, h_dx, 3 * sizeof(double)));
    CUDA_CHECK_OP(cudaMemcpyToSymbol(d_lambda, h_lambda, 3 * sizeof(double)));
    cudaDeviceSynchronize();
    T.stop();
  
//    symbol<<<1,1>>>();  
}
