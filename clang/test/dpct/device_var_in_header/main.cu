// RUN: dpct --process-all -in-root %S -out-root %T --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/common.dp.hpp --match-full-lines %S/common.cuh
// RUN: FileCheck --input-file %T/f.dp.cpp --match-full-lines %S/f.cu
// RUN: FileCheck --input-file %T/main.dp.cpp --match-full-lines %S/main.cu

#include "common.h"
#include "common.cuh"

//CHECK: dpct::constant_memory<int, 1> arr6(sycl::range<1>(2), {1, 2});
__device__ __constant__ int arr6[2] = {1, 2};
//CHECK: static dpct::constant_memory<int, 1> arr7(sycl::range<1>(2), {1, 2});
static __device__ __constant__ int arr7[2] = {1, 2};

int main()
{
  f<<<1, 1>>>();
  g<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
