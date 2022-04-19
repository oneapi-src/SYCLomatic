// RUN: c2s --format-range=none --usm-level=none -out-root %T/kernel_call_no_arg %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/kernel_call_no_arg/kernel_call_no_arg.dp.cpp

#include <cuda_runtime.h>

#include <cassert>

#define NUM_ELEMENTS 16

__device__ float out[NUM_ELEMENTS];

// CHECK: void kernel1(sycl::nd_item<3> [[ITEM:item_ct1]], float *out) {
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
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  const size_t threads_per_block = NUM_ELEMENTS;

  float buf[NUM_ELEMENTS] = { 0 };
  const size_t mem_size = sizeof(float) * NUM_ELEMENTS;

  cudaMemcpyToSymbol(out, buf, mem_size);

  // CHECK:   q_ct1.submit(
  // CHECK:     [&](sycl::handler &cgh) {
  // CHECK:       auto out_acc_ct1 = out.get_access(cgh);
  // CHECK:       cgh.parallel_for<c2s_kernel_name<class kernel1_{{[a-f0-9]+}}>>(
  // CHECK:         sycl::nd_range<3>(sycl::range<3>(1, 1, threads_per_block), sycl::range<3>(1, 1, threads_per_block)),
  // CHECK:         [=](sycl::nd_item<3> [[ITEM:item_ct1]]) {
  // CHECK:           kernel1([[ITEM]], out_acc_ct1.get_pointer());
  // CHECK:         });
  // CHECK:     });
  kernel1<<<1, threads_per_block>>>();

  // CHECK:   q_ct1.parallel_for<c2s_kernel_name<class kernel2_{{[a-f0-9]+}}>>(
  // CHECK:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](sycl::nd_item<3> [[ITEM:item_ct1]]) {
  // CHECK:           kernel2();
  // CHECK:         });
  kernel2<<<1, 1>>>();

  return 0;
}

