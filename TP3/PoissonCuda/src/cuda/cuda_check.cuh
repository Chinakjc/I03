#ifndef CUDA_CHECK_CUH
#define CUDA_CHECK_CUH

#include <iostream>
#include <cuda.h>
#include <driver_types.h>

#define CUDA_CHECK_OP(X) { \
  cudaError_t _m_cudaStat = X; \
  if(cudaSuccess != _m_cudaStat) { \
    std::cerr << " \nCUDA_ERROR: " << cudaGetErrorString(_m_cudaStat)  \
              << " in file " << __FILE__ <<  " line " << __LINE__  \
              << std::endl; \
    exit(1); \
    } \
}

#define CUDA_CHECK_KERNEL() { \
  CUDA_CHECK_OP(cudaGetLastError()); \
}

#endif

