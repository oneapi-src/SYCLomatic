// FIXME
// UNSUPPORTED: -windows-

// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sharedmem_var_static.dp.cpp

#include <stdio.h>
#define SIZE 64

class TestObject{
public:
  // CHECK: static void run(int *in, int *out, cl::sycl::nd_item<3> item_ct1, dpct::accessor<int, dpct::local, 0> a0) {
  __device__ static void run(int *in, int *out) {
    // CHECK:  // the size of s is static
    __shared__ int a0; // the size of s is static
    a0 = threadIdx.x;
  }
};

// CHECK: void nonTypeTemplateReverse(int *d, int n, cl::sycl::nd_item<3> [[ITEM:item_ct1]], dpct::accessor<int, dpct::local, 1> s) {
// CHECK-NEXT:  // the size of s is dependent on parameter
template <int ArraySize>
__global__ void nonTypeTemplateReverse(int *d, int n) {
  __shared__ int s[2*ArraySize*ArraySize]; // the size of s is dependent on parameter
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
  }
}

// CHECK: void staticReverse(int *d, int n, cl::sycl::nd_item<3> [[ITEM:item_ct1]], dpct::accessor<int, dpct::local, 0> a0, dpct::accessor<int, dpct::local, 1> s) {
__global__ void staticReverse(int *d, int n) {
  const int size = 64;
  // CHECK:  // the size of s is static
  __shared__ int s[size]; // the size of s is static
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
  }
  // CHECK: TestObject::run(d, d, item_ct1, a0);
  TestObject::run(d, d);
}

// CHECK: template<typename TData>
// CHECK-NEXT: void templateReverse(TData *d, TData n, cl::sycl::nd_item<3> [[ITEM:item_ct1]], dpct::accessor<TData, dpct::local, 2> s) {
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
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> d_d_buf_ct0 = dpct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:   size_t d_d_offset_ct0 = d_d_buf_ct0.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       dpct::range<2> s_range_ct1(64/*size * 2*/, 128/*size * 4*/);
  // CHECK-NEXT:       cl::sycl::accessor<T, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> s_acc_ct1(s_range_ct1, cgh);
  // CHECK-NEXT:       auto d_d_acc_ct0 = d_d_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class templateReverse_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, n), cl::sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           T *d_d_ct0 = (T *)(&d_d_acc_ct0[0] + d_d_offset_ct0);
  // CHECK-NEXT:           templateReverse<T>(d_d_ct0, n, item_ct1, dpct::accessor<T, dpct::local, 2>(s_acc_ct1, s_range_ct1));
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
  // CHECK-NEXT:   dpct::buffer_t d_d_buf_ct0 = dpct::get_buffer(d_d);
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       dpct::range<0> a0_range_ct1;
  // CHECK-NEXT:       cl::sycl::accessor<int, 0, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> a0_acc_ct1(cgh);
  // CHECK-NEXT:       dpct::range<1> s_range_ct1(64/*size*/);
  // CHECK-NEXT:       cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> s_acc_ct1(s_range_ct1, cgh);
  // CHECK-NEXT:       auto d_d_acc_ct0 = d_d_buf_ct0.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class staticReverse_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, n), cl::sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           staticReverse((int *)(&d_d_acc_ct0[0]), n, item_ct1, dpct::accessor<int, dpct::local, 0>(a0_acc_ct1, a0_range_ct1), dpct::accessor<int, dpct::local, 1>(s_acc_ct1, s_range_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  staticReverse<<<1, n>>>(d_d, n);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);

  // CHECK: {
  // CHECK-NEXT:   dpct::buffer_t d_d_buf_ct0 = dpct::get_buffer(d_d);
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       dpct::range<2> s_range_ct1(64/*size * 2*/, 128/*size * 4*/);
  // CHECK-NEXT:       cl::sycl::accessor<int, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> s_acc_ct1(s_range_ct1, cgh);
  // CHECK-NEXT:       auto d_d_acc_ct0 = d_d_buf_ct0.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class templateReverse_{{[a-f0-9]+}}, int>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, n), cl::sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           templateReverse<int>((int *)(&d_d_acc_ct0[0]), n, item_ct1, dpct::accessor<int, dpct::local, 2>(s_acc_ct1, s_range_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  templateReverse<int><<<1, n>>>(d_d, n);

  // CHECK: {
  // CHECK-NEXT:   dpct::buffer_t d_d_buf_ct0 = dpct::get_buffer(d_d);
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       dpct::range<1> s_range_ct1(2*SIZE*SIZE);
  // CHECK-NEXT:       cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> s_acc_ct1(s_range_ct1, cgh);
  // CHECK-NEXT:       auto d_d_acc_ct0 = d_d_buf_ct0.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class nonTypeTemplateReverse_{{[a-f0-9]+}}, dpct_kernel_scalar<SIZE>>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, n), cl::sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           nonTypeTemplateReverse<SIZE>((int *)(&d_d_acc_ct0[0]), n, item_ct1, dpct::accessor<int, dpct::local, 1>(s_acc_ct1, s_range_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  nonTypeTemplateReverse<SIZE><<<1, n>>>(d_d, n);
}
