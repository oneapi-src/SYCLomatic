// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/kernel_call_no_arg.dp.cpp

#include <cuda_runtime.h>

#include <cassert>

#define NUM_ELEMENTS 16

__device__ float out[NUM_ELEMENTS];

// CHECK: void kernel1(cl::sycl::nd_item<3> [[ITEM:item_ct1]], dpct::accessor<float, dpct::device, 1> out) {
// CHECK:   out[{{.*}}[[ITEM]].get_local_id(2)] = [[ITEM]].get_local_id(2);
// CHECK: }
__global__ void kernel1() {
  out[threadIdx.x] = threadIdx.x;
}

// CHECK: void kernel2() {
__global__ void kernel2() {
  int a = 2;
}

int main() {
  const size_t threads_per_block = NUM_ELEMENTS;

  float buf[NUM_ELEMENTS] = { 0 };
  const size_t mem_size = sizeof(float) * NUM_ELEMENTS;

  cudaMemcpyToSymbol(out, buf, mem_size);

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto out_acc_ct1 = out.get_access(cgh);
  // CHECK:       cgh.parallel_for<dpct_kernel_name<class kernel1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, threads_per_block), cl::sycl::range<3>(1, 1, threads_per_block)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_ct1]]) {
  // CHECK:           kernel1([[ITEM]], dpct::accessor<float, dpct::device, 1>(out_acc_ct1));
  // CHECK:         });
  // CHECK:     });
  kernel1<<<1, threads_per_block>>>();

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       cgh.parallel_for<dpct_kernel_name<class kernel2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_ct1]]) {
  // CHECK:           kernel2();
  // CHECK:         });
  // CHECK:     });
  kernel2<<<1, 1>>>();

  return 0;
}
