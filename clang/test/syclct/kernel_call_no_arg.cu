// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/kernel_call_no_arg.sycl.cpp

#include <cuda_runtime.h>

#include <cassert>

#define NUM_ELEMENTS 16

__device__ float out[NUM_ELEMENTS];

// CHECK: void kernel1(cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]], syclct::syclct_accessor<float, syclct::device, 1> out) {
// CHECK:   out[{{.*}}[[ITEM]].get_local_id(0)] = [[ITEM]].get_local_id(0);
// CHECK: }
__global__ void kernel1() {
  out[threadIdx.x] = threadIdx.x;
}

// CHECK: void kernel2() {
// CHECK:   printf("Hello World!\n");
// CHECK: }
__global__ void kernel2() {
  printf("Hello World!\n");
}

int main() {
  const size_t threads_per_block = NUM_ELEMENTS;

  float buf[NUM_ELEMENTS] = { 0 };
  const size_t mem_size = sizeof(float) * NUM_ELEMENTS;

  cudaMemcpyToSymbol(out, buf, mem_size);

  // CHECK: {
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto out_acc = out.get_access(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(threads_per_block, 1, 1)), cl::sycl::range<3>(threads_per_block, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           kernel1([[ITEM]], syclct::syclct_accessor<float, syclct::device, 1>(out_acc));
  // CHECK:         });
  // CHECK:     });
  // CHECK: };
  kernel1<<<1, threads_per_block>>>();

  // CHECK: {
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           kernel2();
  // CHECK:         });
  // CHECK:     });
  // CHECK: };
  kernel2<<<1, 1>>>();

  return 0;
}
