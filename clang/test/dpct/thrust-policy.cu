// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: c2s --format-range=none -out-root %T/thrust-policy %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/thrust-policy/thrust-policy.dp.cpp

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>

void checkCopyIf() {
  auto isEven = [] __host__ __device__ (int v) {return (v % 2) == 0;};
  auto isOdd = [](int v) {return (v % 2) != 0;};
  const int N = 4;
  int vecHIn[N] = {-1, 0, 1, 2};
  int vecHOut[N];

  thrust::device_vector<int> dVecIn(vecHIn, vecHIn+N);
  thrust::device_vector<int> dVecOut(N);
 
  // No policy specified.  Assume host memory.
// CHECK:  std::copy_if(oneapi::dpl::execution::seq, vecHIn, vecHIn+N, vecHOut, isEven);
  thrust::copy_if(vecHIn, vecHIn+N, vecHOut, isEven);
 
  int *vecDIn;
  int *vecDOut;
  cudaMalloc(&vecDIn, sizeof(int)*N);
  cudaMalloc(&vecDOut, sizeof(int)*N);
  cudaMemcpy(vecDIn, vecHIn, sizeof(int)*N, cudaMemcpyHostToDevice);
 
  // No policy specified. Assume host memory.  This segfaults with nvcc
// CHECK:  std::copy_if(oneapi::dpl::execution::seq, vecDIn, vecDIn+N, vecDOut, isEven);
  thrust::copy_if(vecDIn, vecDIn+N, vecDOut, isEven);
  cudaMemcpy(vecHOut, vecDOut, sizeof(int)*N, cudaMemcpyDeviceToHost);
 
  // Policy (thrust::device) specified. Derive memory source from policy. This works with nvcc!
// CHECK:  std::copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), vecDIn, vecDIn+N, vecDOut, isEven);
  thrust::copy_if(thrust::device, vecDIn, vecDIn+N, vecDOut, isEven);
  cudaMemcpy(vecHOut, vecDOut, sizeof(int)*N, cudaMemcpyDeviceToHost);

  // Policy (thrust::host) specified. Derive memory source from policy. This works with nvcc!
// CHECK:  std::copy_if(oneapi::dpl::execution::seq, vecHIn, vecHIn+N, vecHOut, isEven);
  thrust::copy_if(thrust::host, vecHIn, vecHIn+N, vecHOut, isEven);

  // No policy specified. thrust::device_vector used for input/output use device policy
// CHECK:  std::copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), dVecIn.begin(), dVecIn.end(), dVecOut.begin(), isEven);
  thrust::copy_if(dVecIn.begin(), dVecIn.end(), dVecOut.begin(), isEven);
}
