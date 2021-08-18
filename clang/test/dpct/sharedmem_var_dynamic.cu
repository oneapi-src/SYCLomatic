// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none --usm-level=none -out-root %T/sharedmem_var_dynamic %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sharedmem_var_dynamic/sharedmem_var_dynamic.dp.cpp

#include <stdio.h>
#define SIZE 100
// CHECK: void staticReverse(int *d, int n, sycl::nd_item<3> item_ct1,
// CHECK-NEXT:               uint8_t *dpct_local) {
// CHECK-NEXT:  auto s = (int *)dpct_local; // the size of s is dynamic
__global__ void staticReverse(int *d, int n) {
  extern __shared__ int s[]; // the size of s is dynamic
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
  }
}

// CHECK: template<typename TData>
// CHECK-NEXT: void templateReverse(TData *d, TData n, sycl::nd_item<3> item_ct1,
// CHECK-NEXT:                      uint8_t *dpct_local) {
template<typename TData>
__global__ void templateReverse(TData *d, TData n) {

  // CHECK: auto s = (TData *)dpct_local; // the size of s is dynamic
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

  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local> dpct_local_acc_ct1(sycl::range<1>(mem_size), cgh);
  // CHECK-NEXT:     dpct::access_wrapper<T *> d_d_acc_ct0(d_d, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class templateReverse_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, n), sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         templateReverse<T>(d_d_acc_ct0.get_raw_pointer(), n, item_ct1, dpct_local_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  templateReverse<T><<<1, n, mem_size>>>(d_d, n);
}

int main(void) {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  const int n = 64;
  int a[n], r[n], d[n];
  int *d_d;
  int mem_size = n * sizeof(int);
  cudaMalloc((void **)&d_d, mem_size);
  cudaMemcpy(d_d, a, mem_size, cudaMemcpyHostToDevice);
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local> dpct_local_acc_ct1(sycl::range<1>(mem_size), cgh);
  // CHECK-NEXT:     auto d_d_acc_ct0 = dpct::get_access(d_d, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class staticReverse_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, n), sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         staticReverse((int *)(&d_d_acc_ct0[0]), n, item_ct1, dpct_local_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  staticReverse<<<1, n, mem_size>>>(d_d, n);
  cudaMemcpy(d, d_d, mem_size, cudaMemcpyDeviceToHost);

  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     /*
  // CHECK-NEXT:     DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  // CHECK-NEXT:     */
  // CHECK-NEXT:     sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local> dpct_local_acc_ct1(sycl::range<1>(sizeof(int)), cgh);
  // CHECK-NEXT:     auto d_d_acc_ct0 = dpct::get_access(d_d, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class staticReverse_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, n), sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         staticReverse((int *)(&d_d_acc_ct0[0]), n, item_ct1, dpct_local_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  staticReverse<<<1, n, sizeof(int)>>>(d_d, n);

  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local> dpct_local_acc_ct1(sycl::range<1>(4), cgh);
  // CHECK-NEXT:     auto d_d_acc_ct0 = dpct::get_access(d_d, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class templateReverse_{{[a-f0-9]+}}, int>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, n), sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         templateReverse<int>((int *)(&d_d_acc_ct0[0]), n, item_ct1, dpct_local_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  templateReverse<int><<<1, n, 4>>>(d_d, n);
}

// CHECK: void foo_1(uint8_t *dpct_local) {
// CHECK-NEXT:  auto shad_mem_1 = (int(*)[2])dpct_local;
// CHECK-NEXT:  int p = shad_mem_1[0][0];
// CHECK-NEXT:}
__global__ void foo_1() {
  extern __shared__ int shad_mem_1[][2];
  int p = shad_mem_1[0][0];
}

// CHECK:void foo_2(uint8_t *dpct_local) {
// CHECK-NEXT:  auto shad_mem_2 = (int(*)[2][3])dpct_local;
// CHECK-NEXT:  int p = shad_mem_2[0][0][2];
// CHECK-NEXT:}
__global__ void foo_2() {
  extern __shared__ int shad_mem_2[][2][3];
  int p = shad_mem_2[0][0][2];
}

// CHECK:void foo_3(uint8_t *dpct_local) {
// CHECK-NEXT:  auto shad_mem_3 = (int(*)[2][3])dpct_local;
// CHECK-NEXT:}
__global__ void foo_3() {
  extern __shared__ int shad_mem_3[][2][3];
}
