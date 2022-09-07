// RUN: cd %T
// RUN: cat %S/compile_commands.json > %T/compile_commands.json
// RUN: cat %S/t2.c > %T/t2.c
// RUN: cat %S/withoutxcuda.c > %T/withoutxcuda.c

// RUN: dpct --format-range=none -in-root=%T  -out-root=%T/out withoutxcuda.c --format-range=none --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %T/withoutxcuda.c --match-full-lines --input-file %T/out/withoutxcuda.c.dp.cpp

// This file is migrated as CUDA file as default if compilation db is not used.

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>

// CHECK: dpct::constant_memory<float, 1> const_angle(360);
// CHECK-NEXT: void simple_kernel(float *d_array, float *const_angle) {
// CHECK-NEXT:   d_array[0] = const_angle[0];
// CHECK-NEXT:   return;
// CHECK-NEXT: }
#ifndef DTEST
__constant__ float const_angle[360];
__global__ void simple_kernel(float *d_array) {
  d_array[0] = const_angle[0];
  return;
}
#else
__constant__ float const_angle[360];
#endif
