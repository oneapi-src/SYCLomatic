// RUN: echo "empty command"

#include "common.h"
#include "common.cuh"
#include <cstdio>

//CHECK: dpct::constant_memory<int, 1> arr4(sycl::range<1>(2), {1, 2});
__device__ __constant__ int arr4[2] = {1, 2};
//CHECK: static dpct::constant_memory<int, 1> arr5(sycl::range<1>(2), {1, 2});
static __device__ __constant__ int arr5[2] = {1, 2};

__global__ void f() {
  printf("%d\n", arr[0]);
}

__global__ void f() {
  printf("%d\n", arr2[0]);
}
