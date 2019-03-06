// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/devicemem.sycl.cpp

#include <cuda_runtime.h>

#include <cassert>

#define NUM_ELEMENTS (/* Threads per block */16)

// TODO:
//   1. Multiple device variables used in a kernel function (usage analysis)
//   2. Initialized value for device variable
//   3. Muti-dimensional array

// CHECK: syclct::device_memory<float, 1> in(16);
__device__ float in[NUM_ELEMENTS];
// CHECK: syclct::device_memory<int, 1> init(syclct::syclct_range<1>(4), {1, 2, 3, 4});
__device__ int init[4] = {1, 2, 3, 4};

// CHECK: void kernel1(float *out, cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]], syclct::syclct_accessor<float, syclct::device, 1> in) {
// CHECK:   out[{{.*}}[[ITEM]].get_local_id(0)] = in[{{.*}}[[ITEM]].get_local_id(0)];
// CHECK: }
__global__ void kernel1(float *out) {
  out[threadIdx.x] = in[threadIdx.x];
}

// CHECK: syclct::device_memory<int, 0> al;
__device__ int al;
// CHECK: syclct::device_memory<int, 0> ainit(syclct::syclct_range<0>(), (16));
__device__ int ainit = NUM_ELEMENTS;

// CHECK: syclct::device_memory<float, 1> fx(2);
// CHECK: syclct::device_memory<float, 2> fy(3, 4);
__device__ float fx[2], fy[3][4];

// CHECK: void kernel2(float *out, cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]], syclct::syclct_accessor<int, syclct::device, 0> al, syclct::syclct_accessor<float, syclct::device, 1> fx, syclct::syclct_accessor<float, syclct::device, 2> fy) {
// CHECK:   out[{{.*}}[[ITEM]].get_local_id(0)] += (int)al;
// CHECK:   fx[{{.*}}[[ITEM]].get_local_id(0)] = fy[{{.*}}[[ITEM]].get_local_id(0)][{{.*}}[[ITEM]].get_local_id(0)];
// CHECK: }
__global__ void kernel2(float *out) {
  out[threadIdx.x] += al;
  fx[threadIdx.x] = fy[threadIdx.x][threadIdx.x];
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
  cudaMemcpyToSymbol(al, &h_a, sizeof(int));

  float *d_out = NULL;
  cudaMalloc((void **)&d_out, array_size);

  const int threads_per_block = NUM_ELEMENTS;
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> d_out_buf = syclct::get_buffer_and_offset(d_out);
  // CHECK:   size_t d_out_offset = d_out_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto in_acc = in.get_access(cgh);
  // CHECK:       auto d_out_acc = d_out_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(threads_per_block, 1, 1)), cl::sycl::range<3>(threads_per_block, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           float *d_out = (float*)(&d_out_acc[0] + d_out_offset);
  // CHECK:           kernel1(d_out, [[ITEM]], syclct::syclct_accessor<float, syclct::device, 1>(in_acc));
  // CHECK:         });
  // CHECK:     });
  // CHECK: };
  kernel1<<<1, threads_per_block>>>(d_out);

  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> d_out_buf = syclct::get_buffer_and_offset(d_out);
  // CHECK:   size_t d_out_offset = d_out_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto al_acc = al.get_access(cgh);
  // CHECK:       auto fx_acc = fx.get_access(cgh);
  // CHECK:       auto fy_acc = fy.get_access(cgh);
  // CHECK:       auto d_out_acc = d_out_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(threads_per_block, 1, 1)), cl::sycl::range<3>(threads_per_block, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           float *d_out = (float*)(&d_out_acc[0] + d_out_offset);
  // CHECK:           kernel2(d_out, [[ITEM]], syclct::syclct_accessor<int, syclct::device, 0>(al_acc), syclct::syclct_accessor<float, syclct::device, 1>(fx_acc), syclct::syclct_accessor<float, syclct::device, 2>(fy_acc));
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
