// RUN: c2s --format-range=none --usm-level=none -out-root %T/group_local_memory %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -use-experimental-features=local-memory-kernel-scope-allocation -- -x cuda --cuda-host-only -fno-delayed-template-parsing
// RUN: FileCheck %s --match-full-lines --input-file %T/group_local_memory/group_local_memory.dp.cpp

#include <stdio.h>
#include <complex>
#define SIZE 64

class TestObject{
public:
  // CHECK: static void run(int *in, int *out, sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:  auto &a0 = *sycl::ext::oneapi::group_local_memory<int>(item_ct1.get_group()); // the size of s is static
  // CHECK-NEXT:  a0 = item_ct1.get_local_id(2);
  __device__ static void run(int *in, int *out) {
    __shared__ int a0; // the size of s is static
    a0 = threadIdx.x;
  }
  __device__ void test() {}
};

// CHECK: void memberAcc(sycl::nd_item<3> item_ct1) {
// CHECK-NEXT: auto &s = *sycl::ext::oneapi::group_local_memory<TestObject>(item_ct1.get_group()); // the size of s is static
// CHECK-NEXT: s.test();
// CHECK-NEXT: }
__global__ void memberAcc() {
  __shared__ TestObject s; // the size of s is static
  s.test();
}

// CHECK: void nonTypeTemplateReverse(int *d, int n, sycl::nd_item<3> [[ITEM:item_ct1]]) {
// CHECK-NEXT:  auto &s = *sycl::ext::oneapi::group_local_memory<std::complex<int>[2*ArraySize*ArraySize]>(item_ct1.get_group()); // the size of s is dependent on parameter
template <int ArraySize>
__global__ void nonTypeTemplateReverse(int *d, int n) {
  __shared__ std::complex<int> s[2*ArraySize*ArraySize]; // the size of s is dependent on parameter
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
  }
}

// CHECK: void staticReverse(int *d, int n, sycl::nd_item<3> [[ITEM:item_ct1]]) {
__global__ void staticReverse(int *d, int n) {
  const int size = 64;
  // CHECK:  auto &s = *sycl::ext::oneapi::group_local_memory<int[size]>(item_ct1.get_group()); // the size of s is static
  __shared__ int s[size]; // the size of s is static
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
  }
  // CHECK: TestObject::run(d, d, item_ct1);
  TestObject::run(d, d);
}

// CHECK: template<typename TData>
// CHECK-NEXT: void templateReverse(TData *d, TData n, sycl::nd_item<3> [[ITEM:item_ct1]]) {
template<typename TData>
__global__ void templateReverse(TData *d, TData n) {
  const int size = 32;
  // CHECK:  auto &s = *sycl::ext::oneapi::group_local_memory<TData[size * 2][size * 4]>(item_ct1.get_group()); // the size of s is static
  // CHECK-NEXT:  auto &s3 = *sycl::ext::oneapi::group_local_memory<TData[size * 2][size * 4][size]>(item_ct1.get_group()); // the size of s is static
  __shared__ TData s[size * 2][size * 4]; // the size of s is static
  __shared__ TData s3[size * 2][size * 4][size]; // the size of s is static
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

  // CHECK: c2s::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     c2s::access_wrapper<T *> d_d_acc_ct0(d_d, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<c2s_kernel_name<class templateReverse_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, n), sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         templateReverse<T>(d_d_acc_ct0.get_raw_pointer(), n, item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  templateReverse<T><<<1, n>>>(d_d, n);
}

int main(void) {
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  const int n = 64;
  int a[n], r[n], d[n];
  int *d_d;
  cudaMalloc((void **)&d_d, n * sizeof(int));
  cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice);


  // CHECK: q_ct1.parallel_for<c2s_kernel_name<class memberAcc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           memberAcc(item_ct1);
  // CHECK-NEXT:         });
  memberAcc<<<1, 1>>>();
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     auto d_d_acc_ct0 = c2s::get_access(d_d, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<c2s_kernel_name<class staticReverse_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, n), sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         staticReverse((int *)(&d_d_acc_ct0[0]), n, item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  staticReverse<<<1, n>>>(d_d, n);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);

  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     auto d_d_acc_ct0 = c2s::get_access(d_d, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<c2s_kernel_name<class templateReverse_{{[a-f0-9]+}}, int>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, n), sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         templateReverse<int>((int *)(&d_d_acc_ct0[0]), n, item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  templateReverse<int><<<1, n>>>(d_d, n);

  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     auto d_d_acc_ct0 = c2s::get_access(d_d, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<c2s_kernel_name<class nonTypeTemplateReverse_{{[a-f0-9]+}}, c2s_kernel_scalar<SIZE>>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, n), sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         nonTypeTemplateReverse<SIZE>((int *)(&d_d_acc_ct0[0]), n, item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  nonTypeTemplateReverse<SIZE><<<1, n>>>(d_d, n);
}

