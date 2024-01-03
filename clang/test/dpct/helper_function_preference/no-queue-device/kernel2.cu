// RUN: echo "empty command"

#include "common.cuh"

__global__ void kernel2(int *d_Data) {}

static uint *d_Data2;

// CHECK: void malloc2() { d_Data2 = (uint *)sycl::malloc_device(SIZE * sizeof(int), q_ct1); }
void malloc2() { cudaMalloc((void **)&d_Data2, SIZE * sizeof(int)); }

// CHECK: void free2() { sycl::free(d_Data2, q_ct1); }
void free2() { cudaFree(d_Data2); }

// CHECK: void kernelWrapper2(int *d_Data) {
// CHECK-NEXT:   q_ct1.parallel_for(
// CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:         kernel2(d_Data);
// CHECK-NEXT:       });
// CHECK-NEXT:   q_ct1.parallel_for(
// CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:         kernel2(d_Data);
// CHECK-NEXT:       });
void kernelWrapper2(int *d_Data) {
  kernel2<<<1, 1>>>(d_Data);
  kernel2<<<1, 1>>>(d_Data);
}
