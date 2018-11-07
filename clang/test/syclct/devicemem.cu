// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/devicemem.sycl.cpp

#include <cuda_runtime.h>

#include <cassert>

#define NUM_ELEMENTS (/* Threads per block */16)

// TODO:
//   1. Multiple device variables used in a kernel function (usage analysis)
//   2. Initialized value for device variable
//   3. Muti-dimensional array

// CHECK: syclct::DeviceMem in(16* sizeof(float));
__device__ float in[NUM_ELEMENTS];

// CHECK: void kernel1(cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]], cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> in, float *out) {
// CHECK:   out[{{.*}}[[ITEM]].get_local_id(0)] = in[{{.*}}[[ITEM]].get_local_id(0)];
// CHECK: }
__global__ void kernel1(float *out) {
  out[threadIdx.x] = in[threadIdx.x];
}

// CHECK: syclct::DeviceMem a(1* sizeof(int));
__device__ int a;

// CHECK: void kernel2(cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]], cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> a, float *out) {
// CHECK:   out[{{.*}}[[ITEM]].get_local_id(0)] += a[0];
// CHECK: }
__global__ void kernel2(float *out) {
  out[threadIdx.x] += a;
}

int main() {
  float h_in[NUM_ELEMENTS] = { 0 };
  float h_out[NUM_ELEMENTS] = { 0 };

  for (int i = 0; i < NUM_ELEMENTS; ++i) {
    h_in[i] = i;
    h_out[i] = -i;
  }

  const size_t array_size = sizeof(float) * NUM_ELEMENTS;
  // CTST-50
  cudaMemcpyToSymbol(in, h_in, array_size);

  const int h_a = 3;
  // CTST-50
  cudaMemcpyToSymbol(a, &h_a, sizeof(int));

  float *d_out = NULL;
  cudaMalloc((void **)&d_out, array_size);

  const int threads_per_block = NUM_ELEMENTS;
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> d_out_buf = syclct::get_buffer_and_offset(d_out);
  // CHECK:   size_t d_out_offset = d_out_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto d_out_acc = d_out_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       auto device_buffer_and_offset_in = syclct::get_buffer_and_offset(in.get_ptr());
  // CHECK:       auto device_buffer_in = device_buffer_and_offset_in.first.reinterpret<float>(cl::sycl::range<1>(16));
  // CHECK:       auto device_acc_in= device_buffer_in.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<SyclKernelName<class kernel1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<1>((cl::sycl::range<1>(1) * cl::sycl::range<1>(threads_per_block)), cl::sycl::range<1>(threads_per_block)),
  // CHECK:         [=](cl::sycl::nd_item<1> it) {
  // CHECK:           float *d_out = (float*)(&d_out_acc[0] + d_out_offset);
  // CHECK:           kernel1(it, device_acc_in, d_out);
  // CHECK:         });
  // CHECK:     });
  // CHECK: };
  kernel1<<<1, threads_per_block>>>(d_out);

  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> d_out_buf = syclct::get_buffer_and_offset(d_out);
  // CHECK:   size_t d_out_offset = d_out_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto d_out_acc = d_out_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       auto device_buffer_and_offset_a = syclct::get_buffer_and_offset(a.get_ptr());
  // CHECK:       auto device_buffer_a = device_buffer_and_offset_a.first.reinterpret<int>(cl::sycl::range<1>(1));
  // CHECK:       auto device_acc_a= device_buffer_a.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<SyclKernelName<class kernel2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<1>((cl::sycl::range<1>(1) * cl::sycl::range<1>(threads_per_block)), cl::sycl::range<1>(threads_per_block)),
  // CHECK:         [=](cl::sycl::nd_item<1> it) {
  // CHECK:           float *d_out = (float*)(&d_out_acc[0] + d_out_offset);
  // CHECK:           kernel2(it, device_acc_a, d_out);
  // CHECK:         });
  // CHECK:     });
  // CHECK: };
  kernel2<<<1, threads_per_block>>>(d_out);

  cudaMemcpy(h_out, d_out, array_size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < NUM_ELEMENTS; ++i) {
    assert(h_out[i] == i + h_a && "Value mis-calculated!");
  }

  return 0;
}
