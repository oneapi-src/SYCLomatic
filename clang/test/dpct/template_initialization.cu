// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
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
// CHECK:   out[{{.*}}[[ITEM]].get_local_id(2)] = in[{{.*}}[[ITEM]].get_local_id(2)];
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
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> d_in_buf_ct0 = dpct::get_buffer_and_offset(d_in);
  // CHECK-NEXT:   size_t d_in_offset_ct0 = d_in_buf_ct0.second;
  // CHECK-NEXT:   dpct::buffer_t d_out_buf_ct1 = dpct::get_buffer(d_out);
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto d_in_acc_ct0 = d_in_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto d_out_acc_ct1 = d_out_buf_ct1.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(num_threads, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(num_threads, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           T *d_in_ct0 = (T *)(&d_in_acc_ct0[0] + d_in_offset_ct0);
  // CHECK-NEXT:           kernel<T>(d_in_ct0, (T *)(&d_out_acc_ct1[0]), item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  kernel<T><<<1, num_threads>>>(d_in, d_out);

  cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < num_threads; ++i) {
    assert(h_out[i] == h_in[i]);
  }
}
