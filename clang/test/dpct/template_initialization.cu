// RUN: dpct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/template_initialization.dp.cpp

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
// CHECK: void kernel(T* in, T* out, cl::sycl::nd_item<3> [[ITEM:item_ct1]]) {
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
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset(d_in);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct1_buf = dpct::get_buffer_and_offset(d_out);
  // CHECK-NEXT:   size_t arg_ct1_offset = arg_ct1_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto arg_ct1_acc = arg_ct1_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(num_threads, 1, 1)), cl::sycl::range<3>(num_threads, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           T *arg_ct0 = (T *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           T *arg_ct1 = (T *)(&arg_ct1_acc[0] + arg_ct1_offset);
  // CHECK-NEXT:           kernel<T>(arg_ct0, arg_ct1, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  kernel<T><<<1, num_threads>>>(d_in, d_out);

  cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < num_threads; ++i) {
    assert(h_out[i] == h_in[i]);
  }
}
