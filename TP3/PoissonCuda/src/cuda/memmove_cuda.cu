#include "memmove.hxx"
#include "cuda_check.cuh"

double * allocate(int n) {
  double *d;
  CUDA_CHECK_OP(cudaMalloc(&d, n*sizeof(double)));
  return d;
}
  
void deallocate(double *&d) {
  CUDA_CHECK_OP(cudaFree(d));
  d = NULL;
}
  
void copyDeviceToHost(double *h, double *d, int n)
{
  cudaMemcpy(h, d, n * sizeof(double), cudaMemcpyDeviceToHost);
}
  
void copyHostToDevice(double *h, double *d, int n)
{
  cudaMemcpy(h, d, n * sizeof(double), cudaMemcpyHostToDevice);
}
  
void copyDeviceToDevice(double *d_out, double *d_in, int n)
{
  cudaMemcpy(d_out, d_in, n * sizeof(double), cudaMemcpyDeviceToDevice);
}
  
  
