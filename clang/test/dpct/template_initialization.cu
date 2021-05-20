// RUN: dpct --format-range=none --usm-level=none -out-root %T/template_initialization %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/template_initialization/template_initialization.dp.cpp

#include <cuda_runtime.h>

#include <cassert>

const int num_threads = 16;

template<typename T>
void run_test();

int main() {
  run_test<float>();
  return 0;
}

template<typename T>
class M{};
// CHECK: void foo(const M<sycl::float2>& in) {
void foo(const M<float2>& in) {
}


// CHECK: template<typename T>
// CHECK: void kernel(T* in, T* out, sycl::nd_item<3> [[ITEM:item_ct1]]) {
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

  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<T *> d_in_acc_ct0(d_in, cgh);
  // CHECK-NEXT:     auto d_out_acc_ct1 = dpct::get_access(d_out, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, num_threads), sycl::range<3>(1, 1, num_threads)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel<T>(d_in_acc_ct0.get_raw_pointer(), (T *)(&d_out_acc_ct1[0]), item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel<T><<<1, num_threads>>>(d_in, d_out);

  cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < num_threads; ++i) {
    assert(h_out[i] == h_in[i]);
  }
}

