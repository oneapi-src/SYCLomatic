#pragma once

#define C10_WARP_SIZE 32
#include <cuda.h>

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = C10_WARP_SIZE, unsigned int mask = 0xffffffff) {
  return __shfl_down_sync(mask, value, delta, width);
}

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T *shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < blockDim.x / C10_WARP_SIZE) ? shared[lid] : T(0);
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}