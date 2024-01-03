// RUN: echo "empty command"

#include "common.cuh"

__global__ void kernel1(int *d_Data) {}

static uint *d_Data1;

// CHECK: void malloc1() { d_Data1 = (uint *)sycl::malloc_device(SIZE * sizeof(int), q_ct1); }
void malloc1() { cudaMalloc((void **)&d_Data1, SIZE * sizeof(int)); }

// CHECK: void free1() { sycl::free(d_Data1, q_ct1); }
void free1() { cudaFree(d_Data1); }

// CHECK: void kernelWrapper1(int *d_Data) {
// CHECK-NEXT:   q_ct1.parallel_for(
// CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:         kernel1(d_Data);
// CHECK-NEXT:       });
// CHECK-NEXT:   q_ct1.parallel_for(
// CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:         kernel1(d_Data);
// CHECK-NEXT:       });
// CHECK-NEXT: }
void kernelWrapper1(int *d_Data) {
  kernel1<<<1, 1>>>(d_Data);
  kernel1<<<1, 1>>>(d_Data);
}
