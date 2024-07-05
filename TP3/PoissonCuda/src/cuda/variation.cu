
#include "cuda_check.cuh"
#include "variation.hxx"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "cuda_check.cuh"
#include "timer.hxx"
#include "memmove.hxx"
#include "values.hxx"

void allocVariationData(double *& diff, int n,
  double *& diffPartial, int nPartial)
{
  if (diff == NULL) 
    diff = allocate(n);
  if (diffPartial == NULL)
    diffPartial = allocate(nPartial);
}

void freeVariationData(double *& diff, 
 double *& diffPartial)
{
  deallocate(diff);
  deallocate(diffPartial);
}

__global__ void
difference (const double *u, 
            const double *v, 
            double * duv, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i<n) {
    duv[i] = fabs(u[i] - v[i]);
  }   
}

__global__ void
reduce(const double *u, double *partialDiff, int n)
{
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ double sdata[];
  
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int i = bid*blockDim.x + tid;

  sdata[tid] = (i<n) ? u[i] : 0.0 ;
  cg::sync(cta);

  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
      if (tid < s)
        sdata[tid] += sdata[tid + s];

      cg::sync(cta);
    }
  cg::sync(cta);
  if (tid == 0) {
    partialDiff[blockIdx.x] = sdata[0];
  }
}

__global__ void printU(const double *u, int n)
{
  int i;
  for (i=0; i<n; i++)
    printf("u[%d] = %g\n", i, u[i]);
}

double variation(const Values &u, 
                        const Values &v,
                        double * &diff,
                        double * &diffPartial, int n)
{
  int dimBlock = 64;
  int dimGrid = int(ceil(n/double(dimBlock)));
  
  int nbytesGrid = dimGrid * sizeof(double);
  int nbytesBlock = dimBlock * sizeof(double);
  int nbytes = n * sizeof(double);

  allocVariationData(diff,        nbytes, 
                     diffPartial, nbytesGrid);

  //printU<<<1,1>>> (u.dataGPU(), n);
  
  difference<<< dimGrid, dimBlock>>> 
     (u.dataGPU(), v.dataGPU(), diff, n); 
  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();

  reduce<<< dimGrid, dimBlock, nbytesBlock >>>
     (diff, diffPartial, n);
  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();

  std::vector<double> h_partialSums(dimGrid);

  copyDeviceToHost(h_partialSums.data(), diffPartial,
                   dimGrid);

  double s = 0.0;
  for (int i=0; i<dimGrid; i++)
    s += h_partialSums[i];

  return s;
}

