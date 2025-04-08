// RUN: dpct --use-experimental-features=device_global -in-root %S -out-root %T/device_global2 %S/device_global2.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/device_global2/device_global2.dp.cpp --match-full-lines %s

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// CHECK: static sycl::ext::oneapi::experimental::device_global<int> var_a;
__device__ int var_a;

template<typename T>
__global__ void kernel(T b) {
  var_a;
}

template __global__ void kernel<int>(int b);
