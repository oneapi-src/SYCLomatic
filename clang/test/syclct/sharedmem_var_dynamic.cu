// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/sharedmem_var_dynamic.sycl.cpp

#include <stdio.h>
#define SIZE 100
// CHECK: void staticReverse(int *d, int n, cl::sycl::nd_item<3> item_{{[a-f0-9]+}}, syclct::syclct_accessor<syclct::byte_t, syclct::shared, 1> syclct_extern_memory) {
// CHECK-NEXT:  auto s = syclct_extern_memory.reinterpret<int>(); // the size of s is dynamic
__global__ void staticReverse(int *d, int n) {
  extern __shared__ int s[]; // the size of s is dynamic
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
    printf("s[%d]=%d\n", t, s[t]);
  }
}

// CHECK: template<typename TData>
// CHECK-NEXT: void templateReverse(TData *d, TData n, cl::sycl::nd_item<3> item_{{[a-f0-9]+}}, syclct::syclct_accessor<syclct::byte_t, syclct::shared, 1> syclct_extern_memory) {
template<typename TData>
__global__ void templateReverse(TData *d, TData n) {

  // CHECK: auto s = syclct_extern_memory.reinterpret<TData>(); // the size of s is dynamic
  extern __shared__ TData s[]; // the size of s is dynamic
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
    printf("s[%d]=%d\n", t, s[t]);
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
  // CHECK-NEXT:  std::pair<syclct::buffer_t, size_t> d_d_buf = syclct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:  size_t d_d_offset = d_d_buf.second;
  // CHECK-NEXT:  syclct::get_default_queue().submit(
  // CHECK-NEXT:	[&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:          auto syclct_extern_memory_acc = syclct::extern_shared_memory(mem_size).get_access(cgh);
  // CHECK-NEXT:	  auto d_d_acc = d_d_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:	  cgh.parallel_for<syclct_kernel_name<class templateReverse_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:		cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(n, 1, 1)), cl::sycl::range<3>(n, 1, 1)),
  // CHECK-NEXT:		[=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:		  T *d_d = (T*)(&d_d_acc[0] + d_d_offset);
  // CHECK-NEXT:		  templateReverse<T>(d_d, n, [[ITEM]], syclct::syclct_accessor<syclct::byte_t, syclct::shared, 1>(syclct_extern_memory_acc));
  // CHECK-NEXT:		});
  // CHECK-NEXT:	});
  // CHECK-NEXT:};
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
  // CHECK-NEXT:  std::pair<syclct::buffer_t, size_t> d_d_buf = syclct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:  size_t d_d_offset = d_d_buf.second;
  // CHECK-NEXT:  syclct::get_default_queue().submit(
  // CHECK-NEXT:	[&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:          auto syclct_extern_memory_acc = syclct::extern_shared_memory(mem_size).get_access(cgh);
  // CHECK-NEXT:	  auto d_d_acc = d_d_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:	  cgh.parallel_for<syclct_kernel_name<class staticReverse_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:		cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(n, 1, 1)), cl::sycl::range<3>(n, 1, 1)),
  // CHECK-NEXT:		[=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:		  int *d_d = (int*)(&d_d_acc[0] + d_d_offset);
  // CHECK-NEXT:		  staticReverse(d_d, n, [[ITEM]], syclct::syclct_accessor<syclct::byte_t, syclct::shared, 1>(syclct_extern_memory_acc));
  // CHECK-NEXT:		});
  // CHECK-NEXT:	});
  // CHECK-NEXT:};
  staticReverse<<<1, n, mem_size>>>(d_d, n);
  cudaMemcpy(d, d_d, mem_size, cudaMemcpyDeviceToHost);

  // CHECK: {
  // CHECK-NEXT:  std::pair<syclct::buffer_t, size_t> d_d_buf = syclct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:  size_t d_d_offset = d_d_buf.second;
  // CHECK-NEXT:  syclct::get_default_queue().submit(
  // CHECK-NEXT:        [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:          auto syclct_extern_memory_acc = syclct::extern_shared_memory(sizeof(int)).get_access(cgh);
  // CHECK-NEXT:          auto d_d_acc = d_d_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:          cgh.parallel_for<syclct_kernel_name<class staticReverse_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:                cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(n, 1, 1)), cl::sycl::range<3>(n, 1, 1)),
  // CHECK-NEXT:                [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:                  int *d_d = (int*)(&d_d_acc[0] + d_d_offset);
  // CHECK-NEXT:                  staticReverse(d_d, n, [[ITEM]], syclct::syclct_accessor<syclct::byte_t, syclct::shared, 1>(syclct_extern_memory_acc));
  // CHECK-NEXT:                });
  // CHECK-NEXT:        });
  // CHECK-NEXT:};
  staticReverse<<<1, n, sizeof(int)>>>(d_d, n);

  // CHECK: {
  // CHECK-NEXT:  std::pair<syclct::buffer_t, size_t> d_d_buf = syclct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:  size_t d_d_offset = d_d_buf.second;
  // CHECK-NEXT:  syclct::get_default_queue().submit(
  // CHECK-NEXT:        [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:          auto syclct_extern_memory_acc = syclct::extern_shared_memory(4).get_access(cgh);
  // CHECK-NEXT:          auto d_d_acc = d_d_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:          cgh.parallel_for<syclct_kernel_name<class templateReverse_{{[a-f0-9]+}}, int>>(
  // CHECK-NEXT:                cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(n, 1, 1)), cl::sycl::range<3>(n, 1, 1)),
  // CHECK-NEXT:                [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:                  int *d_d = (int*)(&d_d_acc[0] + d_d_offset);
  // CHECK-NEXT:                  templateReverse<int>(d_d, n, [[ITEM]], syclct::syclct_accessor<syclct::byte_t, syclct::shared, 1>(syclct_extern_memory_acc));
  // CHECK-NEXT:                });
  // CHECK-NEXT:        });
  // CHECK-NEXT:};
  templateReverse<int><<<1, n, 4>>>(d_d, n);
}

