// FIXME
// UNSUPPORTED: -windows-

// RUN: dpct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/sharedmem_var_static.dp.cpp

#include <stdio.h>
#define SIZE 64
// CHECK: void nonTypeTemplateReverse(int *d, int n, cl::sycl::nd_item<3> [[ITEM:item_ct1]], dpct::dpct_accessor<int, dpct::shared, 1> s) {
// CHECK-NEXT:  // the size of s is dependent on parameter
template <int ArraySize>
__global__ void nonTypeTemplateReverse(int *d, int n) {
  __shared__ int s[2*ArraySize*ArraySize]; // the size of s is dependent on parameter
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
  }
}

// CHECK: void staticReverse(int *d, int n, cl::sycl::nd_item<3> [[ITEM:item_ct1]], dpct::dpct_accessor<int, dpct::shared, 1> s) {
__global__ void staticReverse(int *d, int n) {
  const int size = 64;
  // CHECK:  // the size of s is static
  __shared__ int s[size]; // the size of s is static
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
  }
}

// CHECK: template<typename TData>
// CHECK-NEXT: void templateReverse(TData *d, TData n, cl::sycl::nd_item<3> [[ITEM:item_ct1]], dpct::dpct_accessor<TData, dpct::shared, 2> s) {
template<typename TData>
__global__ void templateReverse(TData *d, TData n) {
  const int size = 32;
  // CHECK:  // the size of s is static
  __shared__ TData s[size * 2][size * 4]; // the size of s is static
  int t = threadIdx.x;
  if (t < 64) {
    s[t][0] = d[t];
  }
}

template <typename T>
void testTemplate() {
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
  // CHECK-NEXT:       dpct::shared_memory<T, 2> s(64/*size * 2*/, 128/*size * 4*/);
  // CHECK-NEXT:       auto s_range_ct1 = s.get_range();
  // CHECK-NEXT:       auto s_acc_ct1 = s.get_access(cgh);
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class templateReverse_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(n, 1, 1)), cl::sycl::range<3>(n, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           T *arg_ct0 = (T *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           templateReverse<T>(arg_ct0, n, item_ct1, dpct::dpct_accessor<T, dpct::shared, 2>(s_acc_ct1, s_range_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  templateReverse<T><<<1, n>>>(d_d, n);
}

int main(void) {
  const int n = 64;
  int a[n], r[n], d[n];
  int *d_d;
  cudaMalloc((void **)&d_d, n * sizeof(int));
  cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice);
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       dpct::shared_memory<int, 1> s(64/*size*/);
  // CHECK-NEXT:       auto s_range_ct1 = s.get_range();
  // CHECK-NEXT:       auto s_acc_ct1 = s.get_access(cgh);
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class staticReverse_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(n, 1, 1)), cl::sycl::range<3>(n, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           int *arg_ct0 = (int *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           staticReverse(arg_ct0, n, item_ct1, dpct::dpct_accessor<int, dpct::shared, 1>(s_acc_ct1, s_range_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  staticReverse<<<1, n>>>(d_d, n);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       dpct::shared_memory<int, 2> s(64/*size * 2*/, 128/*size * 4*/);
  // CHECK-NEXT:       auto s_range_ct1 = s.get_range();
  // CHECK-NEXT:       auto s_acc_ct1 = s.get_access(cgh);
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class templateReverse_{{[a-f0-9]+}}, int>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(n, 1, 1)), cl::sycl::range<3>(n, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           int *arg_ct0 = (int *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           templateReverse<int>(arg_ct0, n, item_ct1, dpct::dpct_accessor<int, dpct::shared, 2>(s_acc_ct1, s_range_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  templateReverse<int><<<1, n>>>(d_d, n);

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       dpct::shared_memory<int, 1> s(2*SIZE*SIZE);
  // CHECK-NEXT:       auto s_range_ct1 = s.get_range();
  // CHECK-NEXT:       auto s_acc_ct1 = s.get_access(cgh);
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class nonTypeTemplateReverse_{{[a-f0-9]+}}, dpct_kernel_scalar<SIZE>>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(n, 1, 1)), cl::sycl::range<3>(n, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           int *arg_ct0 = (int *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           nonTypeTemplateReverse<SIZE>(arg_ct0, n, item_ct1, dpct::dpct_accessor<int, dpct::shared, 1>(s_acc_ct1, s_range_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  nonTypeTemplateReverse<SIZE><<<1, n>>>(d_d, n);
}
