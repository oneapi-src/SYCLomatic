// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/withoutexplicitxcuda.c.dp.cpp

// This file is migrated as CUDA file as default if compilation db is not used.

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

// CHECK: void k(){}
// CHECK-NEXT: int main(int argc, char** argv) {
// CHECK-NEXT:   const int N = 4;
// CHECK-NEXT:   dpct::get_default_queue().submit(
// CHECK-NEXT:     [&](sycl::handler &cgh) {
// CHECK-NEXT:       cgh.parallel_for(
// CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:           k();
// CHECK-NEXT:         });
// CHECK-NEXT:     });
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
__global__ void k(){}
int main(int argc, char** argv) {
  const int N = 4;
  k<<<1,1>>>();
  return 0;
}