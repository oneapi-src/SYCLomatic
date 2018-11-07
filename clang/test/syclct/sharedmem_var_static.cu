// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/sharedmem_var_static.sycl.cpp

#include <stdio.h>
#define SIZE 100
// CHECK: void staticReverse(cl::sycl::nd_item<3> item_{{[a-f0-9]+}}, cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> s, int *d, int n) {
// CHECK-NEXT:  // the size of s is static
__global__ void staticReverse(int *d, int n) {
  __shared__ int s[64]; // the size of s is static
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
    printf("s[%d]=%d\n", t, s[t]);
  }
}

// CHECK: template<typename TData>
// CHECK-NEXT: void templateReverse(cl::sycl::nd_item<3> item_{{[a-f0-9]+}}, cl::sycl::accessor<TData, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> s, TData *d, TData n) {
template <class TData>
__global__ void templateReverse(TData *d, TData n) {
  __shared__ TData s[64][128]; // the size of s is static
  int t = threadIdx.x;
  if (t < 64) {
    s[t][0] = d[t];
    printf("s[%d][0]=%d\n", t, s[t][0]);
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
  // CHECK-NEXT:  std::pair<syclct::buffer_t, size_t> d_d_buf = syclct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:  size_t d_d_offset = d_d_buf.second;
  // CHECK-NEXT:  syclct::get_default_queue().submit(
  // CHECK-NEXT:	[&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:	  auto d_d_acc = d_d_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:	  cl::sycl::accessor<T, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> s(cl::sycl::range<2>(64, 128), cgh);
  // CHECK-NEXT:	  cgh.parallel_for<SyclKernelName<class templateReverse_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:		cl::sycl::nd_range<1>((cl::sycl::range<1>(1) * cl::sycl::range<1>(n)), cl::sycl::range<1>(n)),
  // CHECK-NEXT:		[=](cl::sycl::nd_item<1> it) {
  // CHECK-NEXT:		  T *d_d = (T*)(&d_d_acc[0] + d_d_offset);
  // CHECK-NEXT:		  templateReverse<T>(it, s, d_d, n);
  // CHECK-NEXT:		});
  // CHECK-NEXT:	});
  // CHECK-NEXT:};
  templateReverse<T><<<1, n>>>(d_d, n);
}

int main(void) {
  const int n = 64;
  int a[n], r[n], d[n];
  int *d_d;
  cudaMalloc((void **)&d_d, n * sizeof(int));
  cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice);
  // CHECK: {
  // CHECK-NEXT:  std::pair<syclct::buffer_t, size_t> d_d_buf = syclct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:  size_t d_d_offset = d_d_buf.second;
  // CHECK-NEXT:  syclct::get_default_queue().submit(
  // CHECK-NEXT:	[&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:	  auto d_d_acc = d_d_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:	  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> s(cl::sycl::range<1>(64), cgh);
  // CHECK-NEXT:	  cgh.parallel_for<SyclKernelName<class staticReverse_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:		cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(n, 1, 1)), cl::sycl::range<3>(n, 1, 1)),
  // CHECK-NEXT:		[=](cl::sycl::nd_item<3> it) {
  // CHECK-NEXT:		  int *d_d = (int*)(&d_d_acc[0] + d_d_offset);
  // CHECK-NEXT:		  staticReverse(it, s, d_d, n);
  // CHECK-NEXT:		});
  // CHECK-NEXT:	});
  // CHECK-NEXT:};
  staticReverse<<<1, n>>>(d_d, n);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);

  // CHECK: {
  // CHECK-NEXT:  std::pair<syclct::buffer_t, size_t> d_d_buf = syclct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:  size_t d_d_offset = d_d_buf.second;
  // CHECK-NEXT:  syclct::get_default_queue().submit(
  // CHECK-NEXT:	[&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:	  auto d_d_acc = d_d_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:	  cl::sycl::accessor<int, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> s(cl::sycl::range<2>(64, 128), cgh);
  // CHECK-NEXT:	  cgh.parallel_for<SyclKernelName<class templateReverse_{{[a-f0-9]+}}, int>>(
  // CHECK-NEXT:		cl::sycl::nd_range<1>((cl::sycl::range<1>(1) * cl::sycl::range<1>(n)), cl::sycl::range<1>(n)),
  // CHECK-NEXT:		[=](cl::sycl::nd_item<1> it) {
  // CHECK-NEXT:		  int *d_d = (int*)(&d_d_acc[0] + d_d_offset);
  // CHECK-NEXT:		  templateReverse<int>(it, s, d_d, n);
  // CHECK-NEXT:		});
  // CHECK-NEXT:	});
  // CHECK-NEXT:};
  templateReverse<int><<<1, n>>>(d_d, n);
}
