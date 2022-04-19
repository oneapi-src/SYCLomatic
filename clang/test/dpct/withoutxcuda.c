// RUN: dpct --format-range=none -out-root %T/withoutxcud %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/withoutxcud/withoutxcuda.c.dp.cpp

// This file is migrated as CUDA file as defualt if compilation db is not used.

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>

// CHECK: dpct::constant_memory<float, 1> const_angle(360);
// CHECK-NEXT: void simple_kernel(float *d_array, float *const_angle) {
// CHECK-NEXT:   d_array[0] = const_angle[0];
// CHECK-NEXT:   return;
// CHECK-NEXT: }
__constant__ float const_angle[360];
__global__ void simple_kernel(float *d_array) {
  d_array[0] = const_angle[0];
  return;
}
