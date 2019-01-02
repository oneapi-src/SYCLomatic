// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/template_initialization.sycl.cpp

#include <cuda_runtime.h>

#include <cassert>

const int num_threads = 16;

template<typename T>
void run_test();

int main() {
  run_test<float>();
  return 0;
}

// CHECK: template<typename T>
// CHECK: void kernel(T* in, T* out, cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
// CHECK:   out[{{.*}}[[ITEM]].get_local_id(0)] = in[{{.*}}[[ITEM]].get_local_id(0)];
// CHECK: }
template<typename T>
__global__ void kernel(T* in, T* out) {
  out[threadIdx.x] = in[threadIdx.x];
}

template<typename T>
void run_test() {
  const size_t mem_size = sizeof(T) * num_threads;

  T h_in[num_threads];
  for (int i = 0; i < num_threads; ++i) {
    h_in[i] = (T)i;
  }

  T h_out[num_threads] = { 0 };

  T* d_in;
  cudaMalloc((void **)&d_in, mem_size);
  cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

  T* d_out;
  cudaMalloc((void **)&d_out, mem_size);

  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> d_in_buf = syclct::get_buffer_and_offset(d_in);
  // CHECK:   size_t d_in_offset = d_in_buf.second;
  // CHECK:   std::pair<syclct::buffer_t, size_t> d_out_buf = syclct::get_buffer_and_offset(d_out);
  // CHECK:   size_t d_out_offset = d_out_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto d_in_acc = d_in_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       auto d_out_acc = d_out_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_{{[a-f0-9]+}}, T>>(
  // CHECK:         cl::sycl::nd_range<3>((1 * num_threads), num_threads),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           T *d_in = (T*)(&d_in_acc[0] + d_in_offset);
  // CHECK:           T *d_out = (T*)(&d_out_acc[0] + d_out_offset);
  // CHECK:           kernel<T>(d_in, d_out, [[ITEM]]);
  // CHECK:         });
  // CHECK:     });
  // CHECK: };
  kernel<T><<<1, num_threads>>>(d_in, d_out);

  cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < num_threads; ++i) {
    assert(h_out[i] == h_in[i]);
  }
}
