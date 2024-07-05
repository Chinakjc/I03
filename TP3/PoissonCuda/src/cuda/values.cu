#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <cuda.h>

#include "cuda_check.cuh"
#include "dim.cuh"
#include "user.cuh"

#include "values.hxx"
#include "os.hxx"

__global__ void dprint(double *u)
{
  int i, j, k;
  int p;

  for (k=0; k<d_n[2]; k++) {
    for (j=0; j<d_n[1]; j++) {
      for (i=0; i<d_n[0]; i++) {
        p = i + d_n[0] * (j + d_n[1]*k);
        printf(" %f", u[p]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

__global__
void zeroValue(double *u, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i<n) {
    u[i] = 0.0;
  }
}

void zero(double *d, int n)
{
  int dimBlock = 256;
  int dimGrid = (n + dimBlock - 1)/dimBlock;

  zeroValue<<<dimGrid, dimBlock>>>(d, n);
  CUDA_CHECK_KERNEL();
}


__global__
void initValue(double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int p;

  if (i>0 && i<d_n[0]-1 && j>0 && j<d_n[1]-1 && k>0 && k<d_n[2]-1)
  {
    p = i + d_n[0] * (j + k*d_n[1]);
    u[p] = cond_ini(d_xmin[0] + i*d_dx[0],
                    d_xmin[1] + j*d_dx[1], 
                    d_xmin[2] + k*d_dx[2]);
  }
}

void init(double *d, int n[3])
{
  dim3 dimBlock(8,8,8);
  dim3 dimGrid(int(ceil(n[0]/double(dimBlock.x))),
               int(ceil(n[1]/double(dimBlock.y))),
               int(ceil(n[2]/double(dimBlock.z))));

  initValue<<<dimGrid, dimBlock>>>(d);
  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();
}

__global__
void boundZValue(int k, double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int p;

  if (i<d_n[0] && j<d_n[1]) {
    p = i + j*d_n[0] + k*d_n[0]*d_n[1];
    u[p] = cond_ini(d_xmin[0] + i*d_dx[0], 
                    d_xmin[1] + j*d_dx[1], 
                    d_xmin[2] + k*d_dx[2]);    
  }
}


__global__
void boundYValue(int j, double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int p;

  if (i<d_n[0] && k<d_n[2]) {
    p = i + j*d_n[0] + k*d_n[0]*d_n[1];
    u[p] = cond_ini(d_xmin[0] + i*d_dx[0], 
                    d_xmin[1] + j*d_dx[1], 
                    d_xmin[2] + k*d_dx[2]);
  }
}

__global__
void boundXValue(int i, double *u)
{
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int p;

  if (j<d_n[1] && k<d_n[2]) {
    p = i + j*d_n[0] + k*d_n[0]*d_n[1];
    u[p] = cond_ini(d_xmin[0] + i*d_dx[0], 
                    d_xmin[1] + j*d_dx[1], 
                    d_xmin[2] + k*d_dx[2]);
  }
}

void boundaries(double *d, int n[3], int imin[3], int imax[3])
{
  dim3 dimBlock2(16,16,1);
  dim3 dimGrid2(int(ceil(n[0]/double(dimBlock2.x))),
                int(ceil(n[1]/double(dimBlock2.y))), 
                1);
  boundZValue<<<dimGrid2, dimBlock2>>>(imin[2]-1, d);

  boundZValue<<<dimGrid2, dimBlock2>>>(imax[2]+1, d);

  dim3 dimBlock1(16,1,16);
  dim3 dimGrid1(int(ceil(n[0]/double(dimBlock1.x))), 
                1,
                int(ceil(n[2]/double(dimBlock1.z))));

  boundYValue<<<dimGrid1, dimBlock1>>>(imin[1]-1, d);
  boundYValue<<<dimGrid1, dimBlock1>>>(imax[1]+1, d);

  dim3 dimBlock0(1,16,16);
  dim3 dimGrid0(1, 
                int(ceil(n[1]/double(dimBlock0.y))),
                int(ceil(n[2]/double(dimBlock0.z))));

  boundXValue<<<dimGrid0, dimBlock0>>>(imin[0]-1, d);
  boundXValue<<<dimGrid0, dimBlock0>>>(imax[0]+1, d);

  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();
}
