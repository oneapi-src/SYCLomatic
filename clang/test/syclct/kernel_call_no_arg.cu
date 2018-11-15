// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/kernel_call_no_arg.sycl.cpp

#include <cuda_runtime.h>

#include <cassert>

#define NUM_ELEMENTS 16

__device__ float out[NUM_ELEMENTS];

// CHECK: void kernel1(cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]], cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> out) {
// CHECK:   out[{{.*}}[[ITEM]].get_local_id(0)] = [[ITEM]].get_local_id(0);
// CHECK: }
__global__ void kernel1() {
  out[threadIdx.x] = threadIdx.x;
}

// CHECK: void kernel2(cl::sycl::nd_item<3> item_{{[a-f0-9]+}}) {
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
  // CHECK:       auto device_buffer_and_offset_out = syclct::get_buffer_and_offset(out.get_ptr());
  // CHECK:       auto device_buffer_out = device_buffer_and_offset_out.first.reinterpret<float>(cl::sycl::range<1>(16));
  // CHECK:       auto device_acc_out = device_buffer_out.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<SyclKernelName<class kernel1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(threads_per_block, 1, 1)), cl::sycl::range<3>(threads_per_block, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> it) {
  // CHECK:           kernel1(it, device_acc_out);
  // CHECK:         });
  // CHECK:     });
  // CHECK: };
  kernel1<<<1, threads_per_block>>>();

  // CHECK: {
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       cgh.parallel_for<SyclKernelName<class kernel2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> it) {
  // CHECK:           kernel2(it);
  // CHECK:         });
  // CHECK:     });
  // CHECK: };
  kernel2<<<1, 1>>>();

  return 0;
}
