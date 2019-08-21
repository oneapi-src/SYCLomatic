// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/sharedmem_var_dynamic.dp.cpp

#include <stdio.h>
#define SIZE 100
// CHECK: void staticReverse(int *d, int n, cl::sycl::nd_item<3> item_ct1, dpct::dpct_accessor<dpct::byte_t, dpct::local, 1> dpct_extern_memory) {
// CHECK-NEXT:  auto s = dpct_extern_memory.reinterpret<int>(); // the size of s is dynamic
__global__ void staticReverse(int *d, int n) {
  extern __shared__ int s[]; // the size of s is dynamic
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
  }
}

// CHECK: template<typename TData>
// CHECK-NEXT: void templateReverse(TData *d, TData n, cl::sycl::nd_item<3> item_ct1, dpct::dpct_accessor<dpct::byte_t, dpct::local, 1> dpct_extern_memory) {
template<typename TData>
__global__ void templateReverse(TData *d, TData n) {

  // CHECK: auto s = dpct_extern_memory.reinterpret<TData>(); // the size of s is dynamic
  extern __shared__ TData s[]; // the size of s is dynamic
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
  }
}

template<typename T>
void testTemplate(){
  const int n = 64;
  T a[n], r[n], d[n];
  T *d_d;
  int mem_size = n * sizeof(T);
  cudaMalloc((void **)&d_d, mem_size);
  cudaMemcpy(d_d, a, mem_size, cudaMemcpyHostToDevice);

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       dpct::extern_local_memory dpct_extern_memory(mem_size);
  // CHECK-NEXT:       auto dpct_extern_memory_range_ct1 = dpct_extern_memory.get_range();
  // CHECK-NEXT:       auto dpct_extern_memory_acc_ct1 = dpct_extern_memory.get_access(cgh);
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class templateReverse_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(n, 1, 1)), cl::sycl::range<3>(n, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           T *arg_ct0 = (T *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           templateReverse<T>(arg_ct0, n, item_ct1, dpct::dpct_accessor<dpct::byte_t, dpct::local, 1>(dpct_extern_memory_acc_ct1, dpct_extern_memory_range_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  templateReverse<T><<<1, n, mem_size>>>(d_d, n);
}

int main(void) {
  const int n = 64;
  int a[n], r[n], d[n];
  int *d_d;
  int mem_size = n * sizeof(int);
  cudaMalloc((void **)&d_d, mem_size);
  cudaMemcpy(d_d, a, mem_size, cudaMemcpyHostToDevice);
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       dpct::extern_local_memory dpct_extern_memory(mem_size);
  // CHECK-NEXT:       auto dpct_extern_memory_range_ct1 = dpct_extern_memory.get_range();
  // CHECK-NEXT:       auto dpct_extern_memory_acc_ct1 = dpct_extern_memory.get_access(cgh);
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class staticReverse_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(n, 1, 1)), cl::sycl::range<3>(n, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           int *arg_ct0 = (int *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           staticReverse(arg_ct0, n, item_ct1, dpct::dpct_accessor<dpct::byte_t, dpct::local, 1>(dpct_extern_memory_acc_ct1, dpct_extern_memory_range_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  staticReverse<<<1, n, mem_size>>>(d_d, n);
  cudaMemcpy(d, d_d, mem_size, cudaMemcpyDeviceToHost);

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       dpct::extern_local_memory dpct_extern_memory(sizeof(int));
  // CHECK-NEXT:       auto dpct_extern_memory_range_ct1 = dpct_extern_memory.get_range();
  // CHECK-NEXT:       auto dpct_extern_memory_acc_ct1 = dpct_extern_memory.get_access(cgh);
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class staticReverse_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(n, 1, 1)), cl::sycl::range<3>(n, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           int *arg_ct0 = (int *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           staticReverse(arg_ct0, n, item_ct1, dpct::dpct_accessor<dpct::byte_t, dpct::local, 1>(dpct_extern_memory_acc_ct1, dpct_extern_memory_range_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  staticReverse<<<1, n, sizeof(int)>>>(d_d, n);

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       dpct::extern_local_memory dpct_extern_memory(4);
  // CHECK-NEXT:       auto dpct_extern_memory_range_ct1 = dpct_extern_memory.get_range();
  // CHECK-NEXT:       auto dpct_extern_memory_acc_ct1 = dpct_extern_memory.get_access(cgh);
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class templateReverse_{{[a-f0-9]+}}, int>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(n, 1, 1)), cl::sycl::range<3>(n, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           int *arg_ct0 = (int *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           templateReverse<int>(arg_ct0, n, item_ct1, dpct::dpct_accessor<dpct::byte_t, dpct::local, 1>(dpct_extern_memory_acc_ct1, dpct_extern_memory_range_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  templateReverse<int><<<1, n, 4>>>(d_d, n);
}

