// RUN: dpct --format-range=none --use-experimental-features=device_global -in-root %S -out-root %T/device_global2 %S/device_global2.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/device_global2/device_global2.dp.cpp --match-full-lines %s

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// CHECK: static sycl::ext::oneapi::experimental::device_global<int> var_a;
__device__ int var_a;

// CHECK: static constexpr sycl::ext::oneapi::experimental::device_global<int8_t[2]> var_b  {-1, -1};
static constexpr __device__ int8_t var_b[2] = {-1, -1};

template<typename T>
__global__ void kernel(T b) {
  var_a;
  var_b[0];
}

template __global__ void kernel<int>(int b);
