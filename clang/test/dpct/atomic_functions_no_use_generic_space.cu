// RUN: dpct --format-range=none --usm-level=none --no-use-generic-space=true -out-root %T/atomic_functions_no_use_generic_space %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/atomic_functions_no_use_generic_space/atomic_functions_no_use_generic_space.dp.cpp --match-full-lines %s

#include <cuda_runtime.h>

#include <iostream>
#include <memory>

#define NUM_ATOMICS 11

template <typename T>
__global__ void test(T *data) {
  // CHECK: T tid = item_ct1.get_local_id(2);
  T tid = threadIdx.x;

  // CHECK: dpct::atomic_fetch_add(&data[0], tid);
  atomicAdd(&data[0], tid);

  // CHECK: dpct::atomic_fetch_sub(&data[1], tid);
  atomicSub(&data[1], tid);

  // CHECK: dpct::atomic_exchange(&data[2], tid);
  atomicExch(&data[2], tid);

  // CHECK: dpct::atomic_fetch_max(&data[3], tid);
  atomicMax(&data[3], tid);

  // CHECK: dpct::atomic_fetch_min(&data[4], tid);
  atomicMin(&data[4], tid);

  // CHECK:  dpct::atomic_fetch_compare_inc((unsigned int *)&data[5], (unsigned int)tid);
  atomicInc((unsigned int *)&data[5], (unsigned int)tid);

  // CHECK: /*
  // CHECK: DPCT1007:0: Migration of atomicDec is not supported.
  // CHECK: */
  atomicDec((unsigned int *)&data[6], (unsigned int)tid);

  // CHECK: dpct::atomic_compare_exchange_strong(&data[7], tid - 1, tid);
  atomicCAS(&data[7], tid - 1, tid);

  T old, expected, desired;
  old = data[7];
  do {
    expected = old;
    // CHECK: old = dpct::atomic_compare_exchange_strong(&data[7], expected, desired);
    old = atomicCAS(&data[7], expected, desired);
  } while  (expected != old);

  // CHECK: dpct::atomic_fetch_and(&data[8], tid);
  atomicAnd(&data[8], tid);

  // CHECK: dpct::atomic_fetch_or(&data[9], tid);
  atomicOr(&data[9], tid);

  // CHECK: dpct::atomic_fetch_xor(&data[10], tid);
  atomicXor(&data[10], tid);
}

template <>
__global__ void test(unsigned long long int* data) {
  unsigned long long int tid = threadIdx.x;

  // CHECK: sycl::atomic<unsigned long long>(sycl::global_ptr<unsigned long long>(&data[0])).fetch_add(tid);
  atomicAdd(&data[0], tid);

  // CHECK: dpct::atomic_exchange(&data[2], tid);
  atomicExch(&data[2], tid);

  // CHECK: dpct::atomic_compare_exchange_strong(&data[7], tid - 1, tid);
  atomicCAS(&data[7], tid - 1, tid);
}

template <>
__global__ void test(float* data) {
  float tid = threadIdx.x;

  // CHECK: dpct::atomic_fetch_add(&data[0], tid);
  atomicAdd(&data[0], tid);

  // CHECK: dpct::atomic_exchange(&data[2], tid);
  atomicExch(&data[2], tid);
}

template <>
__global__ void test(double* data) {
  double tid = threadIdx.x;

  // CHECK: dpct::atomic_fetch_add(&data[0], tid);
  atomicAdd(&data[0], tid);
}

template <typename T>
void InvokeKernel() {
  const size_t k_threads_per_block = 1;
  const size_t k_num_elements = NUM_ATOMICS * k_threads_per_block;
  const size_t size = sizeof(T) * k_num_elements;
  std::unique_ptr<T[]> host(new T[k_num_elements]);
  std::fill(host.get(), host.get() + k_num_elements, 0xFF);

  T *dev_ptr;
  cudaMalloc((void **)&dev_ptr, size);

  cudaMemcpy(dev_ptr, host.get(), size, cudaMemcpyHostToDevice);
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<T *> dev_ptr_acc_ct0(dev_ptr, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class test_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, k_threads_per_block), sycl::range<3>(1, 1, k_threads_per_block)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         test<T>(dev_ptr_acc_ct0.get_raw_pointer(), item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  test<T><<<1, k_threads_per_block>>>(dev_ptr);
}

// CHECK: static dpct::global_memory<uint32_t, 1> d_error(1);
static __device__ uint32_t d_error[1];

// CHECK: void fun(uint32_t *d_error){
__device__ void fun(){
  double *a;
  float b;
  // CHECK: dpct::atomic_fetch_add(a, (double)1);
  atomicAdd(a, 1);

  // CHECK: dpct::atomic_fetch_add(a, (double)b);
  atomicAdd(a, b);

  // CHECK: sycl::atomic<uint32_t>(sycl::global_ptr<uint32_t>(d_error)).fetch_add(1);
  atomicAdd(d_error, 1);
}

int main() {
  InvokeKernel<int>();
  InvokeKernel<unsigned int>();
  InvokeKernel<unsigned long long int>();
  InvokeKernel<float>();
  InvokeKernel<double>();
}

// CHECK: void foo(sycl::nd_item<3> item_ct1, uint8_t *dpct_local, uint32_t *share_v) {
// CHECK-NEXT:  auto share_array = (uint32_t *)dpct_local;
// CHECK-NEXT:  for (int b = item_ct1.get_local_id(2); b < 64; b += item_ct1.get_local_range(2)) {
// CHECK-NEXT:    sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(&share_array[b])).fetch_add(1);
// CHECK-NEXT:    sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(share_array)).fetch_add(1);
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(share_v)).fetch_add(1);
// CHECK-NEXT:}
__global__ void foo() {
  extern __shared__ uint32_t share_array[];
  for (int b = threadIdx.x; b < 64; b += blockDim.x) {
    atomicAdd(&share_array[b], 1);
    atomicAdd(share_array, 1);
  }
__shared__ uint32_t share_v;
  atomicAdd(&share_v, 1);
}

// CHECK:void foo_2(sycl::nd_item<3> item_ct1, uint8_t *dpct_local, uint32_t *share_v) {
// CHECK-NEXT:  auto share_array = (uint32_t *)dpct_local;
// CHECK-NEXT:  for (int b = item_ct1.get_local_id(2); b < 64; b += item_ct1.get_local_range(2)) {
// CHECK-NEXT:    uint32_t *p_1 = &share_array[b];
// CHECK-NEXT:    uint32_t *p_2 = share_array;
// CHECK-NEXT:    uint32_t *p_3 = p_2;
// CHECK-NEXT:    sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(p_1)).fetch_add(1);
// CHECK-NEXT:    sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(p_2)).fetch_add(1);
// CHECK-NEXT:    sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(p_3)).fetch_add(1);
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  uint32_t *p_1 = share_v;
// CHECK-NEXT:  uint32_t *p_2 = p_1;
// CHECK-NEXT:  uint32_t *p_3 = p_2;
// CHECK-NEXT:  sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(p_1)).fetch_add(1);
// CHECK-NEXT:  sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(p_2)).fetch_add(1);
// CHECK-NEXT:  sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(p_3)).fetch_add(1);
// CHECK-NEXT:}
__global__ void foo_2() {
  extern __shared__ uint32_t share_array[];
  for (int b = threadIdx.x; b < 64; b += blockDim.x) {
    uint32_t *p_1 = &share_array[b];
    uint32_t *p_2 = share_array;
    uint32_t *p_3 = p_2;
    atomicAdd(p_1, 1);
    atomicAdd(p_2, 1);
    atomicAdd(p_3, 1);
  }
__shared__ uint32_t share_v;
  uint32_t *p_1 = &share_v;
  uint32_t *p_2 = p_1;
  uint32_t *p_3 = p_2;
  atomicAdd(p_1, 1);
  atomicAdd(p_2, 1);
  atomicAdd(p_3, 1);
}

// CHECK:void foo_3(uint32_t *share_v) {
// CHECK-EMPTY:
// CHECK-NEXT:  uint32_t *p_1 = NULL;
// CHECK-NEXT:  uint32_t *p_2 = NULL;
// CHECK-NEXT:  uint32_t *p_3 = NULL;
// CHECK-NEXT:  p_1 = share_v;
// CHECK-NEXT:  p_2 = p_1;
// CHECK-NEXT:  p_3 = p_2;
// CHECK-NEXT:  uint32_t *p_4 = p_3;
// CHECK-NEXT:  sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(p_4)).fetch_add(1);
// CHECK-NEXT:}
__global__ void foo_3() {
__shared__ uint32_t share_v;
  uint32_t *p_1 = NULL;
  uint32_t *p_2 = NULL;
  uint32_t *p_3 = NULL;
  p_1 = &share_v;
  p_2 = p_1;
  p_3 = p_2;
  uint32_t *p_4 = p_3;
  atomicAdd(p_4, 1);
}

// CHECK:dpct::global_memory<uint32_t, 1> dmem(100);
// CHECK-NEXT:void foo_4(uint8_t *dpct_local, uint32_t *dmem) {
// CHECK-NEXT:auto share = (uint32_t *)dpct_local;
// CHECK-NEXT:  uint32_t *p_1 = NULL;
// CHECK-NEXT:  uint32_t *p_2 = NULL;
// CHECK-NEXT:  uint32_t *p_3 = NULL;
// CHECK-NEXT:  p_1 = &share[0];
// CHECK-NEXT:  p_2 = p_1;
// CHECK-NEXT:  p_3 = p_2;
// CHECK-NEXT:  uint32_t *p_4 = p_3;
// CHECK-NEXT:  sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(p_4)).fetch_add(1);
// CHECK-NEXT:  p_1 = share;
// CHECK-NEXT:  p_2 = p_1;
// CHECK-NEXT:  p_3 = p_2;
// CHECK-NEXT:  p_3 = dmem;
// CHECK-NEXT:  uint32_t *p_5 = p_3;
// CHECK-NEXT:  sycl::atomic<uint32_t>(sycl::global_ptr<uint32_t>(p_5)).fetch_add(1);
// CHECK-NEXT:}
__device__ uint32_t dmem[100];
__global__ void foo_4() {
extern __shared__ uint32_t share[];
  uint32_t *p_1 = NULL;
  uint32_t *p_2 = NULL;
  uint32_t *p_3 = NULL;
  p_1 = &share[0];
  p_2 = p_1;
  p_3 = p_2;
  uint32_t *p_4 = p_3;
  atomicAdd(p_4, 1);
  p_1 = share;
  p_2 = p_1;
  p_3 = p_2;
  p_3 = dmem;
  uint32_t *p_5 = p_3;
  atomicAdd(p_5, 1);
}

__device__ uint32_t* func(uint32_t *in){
    return in;
}

// CHECK:void foo_5(uint8_t *dpct_local) {
// CHECK-NEXT:auto share = (uint32_t *)dpct_local;
// CHECK-NEXT:  uint32_t *p_1 = NULL;
// CHECK-NEXT:  uint32_t *p_2 = NULL;
// CHECK-NEXT:  uint32_t *p_3 = NULL;
// CHECK-NEXT:  p_1 = &share[0];
// CHECK-NEXT:  p_2 = p_1;
// CHECK-NEXT:  p_3 = p_2;
// CHECK-NEXT:  uint32_t *p_4;
// CHECK-NEXT:  p_4= p_4 + 1;
// CHECK-NEXT:  sycl::atomic<uint32_t>(sycl::global_ptr<uint32_t>(p_4)).fetch_add(1);
// CHECK-NEXT:  p_4=func(p_3);
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1039:{{[0-9]+}}: The generated code assumes that "p_4" points to the global memory address space. If it points to a local memory address space, replace "sycl::global_ptr" with "sycl::local_ptr".
// CHECK-NEXT:  */
// CHECK-NEXT:  sycl::atomic<uint32_t>(sycl::global_ptr<uint32_t>(p_4)).fetch_add(1);
// CHECK-NEXT:}
__global__ void foo_5() {
extern __shared__ uint32_t share[];
  uint32_t *p_1 = NULL;
  uint32_t *p_2 = NULL;
  uint32_t *p_3 = NULL;
  p_1 = &share[0];
  p_2 = p_1;
  p_3 = p_2;
  uint32_t *p_4;
  p_4= p_4 + 1;
  atomicAdd(p_4, 1);
  p_4=func(p_3);
  atomicAdd(p_4, 1);
}

#define FUNC(in)  in
// CHECK:void foo_6(uint8_t *dpct_local) {
// CHECK-NEXT:auto share = (uint32_t *)dpct_local;
// CHECK-NEXT:  uint32_t *p_1 = NULL;
// CHECK-NEXT:  uint32_t *p_2 = NULL;
// CHECK-NEXT:  uint32_t *p_3 = NULL;
// CHECK-NEXT:  p_1 = &share[0];
// CHECK-NEXT:  p_2 = p_1;
// CHECK-NEXT:  p_3 = p_2;
// CHECK-NEXT:  uint32_t *p_4;
// CHECK-NEXT:  p_4= p_4 + 1;
// CHECK-NEXT:  p_4=FUNC(p_3);
// CHECK-NEXT:  sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(p_4)).fetch_add(1);
// CHECK-NEXT:}
__global__ void foo_6() {
extern __shared__ uint32_t share[];
  uint32_t *p_1 = NULL;
  uint32_t *p_2 = NULL;
  uint32_t *p_3 = NULL;
  p_1 = &share[0];
  p_2 = p_1;
  p_3 = p_2;
  uint32_t *p_4;
  p_4= p_4 + 1;
  p_4=FUNC(p_3);
  atomicAdd(p_4, 1);
}

// CHECK:void foo_7(int a, uint8_t *dpct_local) {
// CHECK-NEXT:auto share = (uint32_t *)dpct_local;
// CHECK-NEXT:  uint32_t *p_1;
// CHECK-NEXT:  uint32_t *p_2;
// CHECK-NEXT:  uint32_t *p_3 = NULL;
// CHECK-NEXT:  p_1 = &share[0];
// CHECK-NEXT:  if(a > 1)
// CHECK-NEXT:    p_2 = p_1;
// CHECK-NEXT:  else
// CHECK-NEXT:    p_2 = p_3;
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1039:{{[0-9]+}}: The generated code assumes that "p_2" points to the global memory address space. If it points to a local memory address space, replace "sycl::global_ptr" with "sycl::local_ptr".
// CHECK-NEXT:  */
// CHECK-NEXT:  sycl::atomic<uint32_t>(sycl::global_ptr<uint32_t>(p_2)).fetch_add(1);
// CHECK-NEXT:}
__global__ void foo_7(int a) {
extern __shared__ uint32_t share[];
  uint32_t *p_1;
  uint32_t *p_2;
  uint32_t *p_3 = NULL;
  p_1 = &share[0];
  if(a > 1)
    p_2 = p_1;
  else
    p_2 = p_3;
  atomicAdd(p_2, 1);
}

#define OFFSET 1

// CHECK:void kernel(unsigned* data, uint8_t *dpct_local) {
// CHECK-NEXT:  auto sm = (unsigned int *)dpct_local;
// CHECK-NEXT:  sm[OFFSET] = data[0];
// CHECK-NEXT:  unsigned* ptr = sm + OFFSET;
// CHECK-NEXT:  sycl::atomic<unsigned int, sycl::access::address_space::local_space>(sycl::local_ptr<unsigned int>(&ptr[0])).fetch_or(data[1]);
// CHECK-NEXT:  data[1] = ptr[0];
// CHECK-NEXT:}
__global__ void kernel(unsigned* data) {
  extern __shared__ unsigned int sm[];
  sm[OFFSET] = data[0];
  unsigned* ptr = sm + OFFSET;
  atomicOr(&ptr[0], data[1]);
  data[1] = ptr[0];
}

// CHECK:void kernel_1(unsigned* data, uint8_t *dpct_local) {
// CHECK-NEXT:  auto sm = (unsigned int *)dpct_local;
// CHECK-NEXT:  sm[OFFSET] = data[0];
// CHECK-NEXT:  unsigned* ptr = OFFSET + sm;
// CHECK-NEXT:  sycl::atomic<unsigned int, sycl::access::address_space::local_space>(sycl::local_ptr<unsigned int>(&ptr[0])).fetch_or(data[1]);
// CHECK-NEXT:  data[1] = ptr[0];
// CHECK-NEXT:}
__global__ void kernel_1(unsigned* data) {
  extern __shared__ unsigned int sm[];
  sm[OFFSET] = data[0];
  unsigned* ptr = OFFSET + sm;
  atomicOr(&ptr[0], data[1]);
  data[1] = ptr[0];
}

// CHECK:void kernel_2(unsigned* data, uint8_t *dpct_local) {
// CHECK-NEXT:  auto sm = (unsigned int *)dpct_local;
// CHECK-NEXT:  sm[OFFSET] = data[0];
// CHECK-NEXT:  unsigned* ptr = OFFSET + sm + (3 + 4) + 6;
// CHECK-NEXT:  sycl::atomic<unsigned int, sycl::access::address_space::local_space>(sycl::local_ptr<unsigned int>(&ptr[0])).fetch_or(data[1]);
// CHECK-NEXT:  data[1] = ptr[0];
// CHECK-NEXT:}
__global__ void kernel_2(unsigned* data) {
  extern __shared__ unsigned int sm[];
  sm[OFFSET] = data[0];
  unsigned* ptr = OFFSET + sm + (3 + 4) + 6;
  atomicOr(&ptr[0], data[1]);
  data[1] = ptr[0];
}

// CHECK: void k(uint32_t *u32) {
__global__ void k() {
  int i;
  unsigned ui;
  unsigned long long ull;
  float f;
  __shared__ uint32_t u32;

  // CHECK: sycl::atomic<int>(sycl::global_ptr<int>(&i)).fetch_add(i);
  // CHECK-NEXT: sycl::atomic<int>(sycl::global_ptr<int>(&i)).fetch_sub(i);
  // CHECK-NEXT: sycl::atomic<int>(sycl::global_ptr<int>(&i)).fetch_and(i);
  // CHECK-NEXT: sycl::atomic<int>(sycl::global_ptr<int>(&i)).fetch_or(i);
  // CHECK-NEXT: sycl::atomic<int>(sycl::global_ptr<int>(&i)).fetch_xor(i);
  // CHECK-NEXT: sycl::atomic<int>(sycl::global_ptr<int>(&i)).fetch_min(i);
  // CHECK-NEXT: sycl::atomic<int>(sycl::global_ptr<int>(&i)).fetch_max(i);
  atomicAdd(&i, i);
  atomicSub(&i, i);
  atomicAnd(&i, i);
  atomicOr(&i, i);
  atomicXor(&i, i);
  atomicMin(&i, i);
  atomicMax(&i, i);

  // CHECK: sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&ui)).fetch_add(ui);
  // CHECK-NEXT: sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&ui)).fetch_sub(ui);
  // CHECK-NEXT: sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&ui)).fetch_and(ui);
  // CHECK-NEXT: sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&ui)).fetch_or(ui);
  // CHECK-NEXT: sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&ui)).fetch_xor(ui);
  // CHECK-NEXT: sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&ui)).fetch_min(ui);
  // CHECK-NEXT: sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&ui)).fetch_max(ui);
  atomicAdd(&ui, ui);
  atomicSub(&ui, ui);
  atomicAnd(&ui, ui);
  atomicOr(&ui, ui);
  atomicXor(&ui, ui);
  atomicMin(&ui, ui);
  atomicMax(&ui, ui);

  // CHECK: sycl::atomic<unsigned long long>(sycl::global_ptr<unsigned long long>(&ull)).fetch_add(ull);
  // CHECK-NEXT: sycl::atomic<unsigned long long>(sycl::global_ptr<unsigned long long>(&ull)).fetch_and(ull);
  // CHECK-NEXT: sycl::atomic<unsigned long long>(sycl::global_ptr<unsigned long long>(&ull)).fetch_or(ull);
  // CHECK-NEXT: sycl::atomic<unsigned long long>(sycl::global_ptr<unsigned long long>(&ull)).fetch_xor(ull);
  // CHECK-NEXT: sycl::atomic<unsigned long long>(sycl::global_ptr<unsigned long long>(&ull)).fetch_min(ull);
  // CHECK-NEXT: sycl::atomic<unsigned long long>(sycl::global_ptr<unsigned long long>(&ull)).fetch_max(ull);
  atomicAdd(&ull, ull);
  atomicAnd(&ull, ull);
  atomicOr(&ull, ull);
  atomicXor(&ull, ull);
  atomicMin(&ull, ull);
  atomicMax(&ull, ull);

  // CHECK: sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&ui)).fetch_add(i);
  atomicAdd(&ui, i);

  // CHECK: sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&ui)).fetch_add(i + i);
  atomicAdd(&ui, i + i);

  // CHECK: sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(u32)).fetch_add(*u32);
  atomicAdd(&u32, u32);

  // CHECK: dpct::atomic_fetch_add(&f, f);
  atomicAdd(&f, f);
}

// CHECK: void mykernel(unsigned int *dev, sycl::nd_item<3> item_ct1, uint8_t *dpct_local) {
// CHECK-NEXT:  auto sm = (unsigned int *)dpct_local;
// CHECK-NEXT:  unsigned int* as= (unsigned int*)sm;
// CHECK-NEXT:  const int kc=item_ct1.get_local_id(2);
// CHECK-NEXT:  const int tid=item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);
// CHECK-NEXT:  sycl::atomic<unsigned int, sycl::access::address_space::local_space>(sycl::local_ptr<unsigned int>(&as[kc])).fetch_or((unsigned int)1);
// CHECK-NEXT:  dev[tid]=as[kc];
// CHECK-NEXT: }
__global__ void mykernel(unsigned int *dev) {
  extern __shared__ unsigned int sm[];
  unsigned int* as= (unsigned int*)sm;
  const int kc=threadIdx.x;
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  atomicOr(&as[kc], (unsigned int)1);
  dev[tid]=as[kc];
}

// CHECK: void mykernel_1(unsigned char *buffer, long size,
// CHECK-NEXT:                             unsigned int *histo, sycl::nd_item<3> item_ct1,
// CHECK-NEXT:                             unsigned int *temp) {
// CHECK-EMPTY:
// CHECK-NEXT:  temp[item_ct1.get_local_id(2)] = 0;
// CHECK-NEXT:  sycl::group_barrier(item_ct1.get_group());
// CHECK-NEXT:  int i = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
// CHECK-NEXT:  int offset = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
// CHECK-NEXT:  while (i < size) {
// CHECK-NEXT:    sycl::atomic<unsigned int, sycl::access::address_space::local_space>(sycl::local_ptr<unsigned int>(&temp[buffer[i]])).fetch_add(1);
// CHECK-NEXT:    i += offset;
// CHECK-NEXT:  }
// CHECK-NEXT:  sycl::group_barrier(item_ct1.get_group());
// CHECK-NEXT:  sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&(histo[item_ct1.get_local_id(2)]))).fetch_add(temp[item_ct1.get_local_id(2)]);
// CHECK-NEXT:}
__global__ void mykernel_1(unsigned char *buffer, long size,
                             unsigned int *histo) {
__shared__ unsigned int temp[256];
  temp[threadIdx.x] = 0;
  __syncthreads();
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = blockDim.x * gridDim.x;
  while (i < size) {
    atomicAdd(&temp[buffer[i]], 1);
    i += offset;
  }
  __syncthreads();
  atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}

// CHECK: /*
// CHECK-NEXT: DPCT1058:{{[0-9]+}}: "atomicAdd" is not migrated because it is not called in the code.
// CHECK-NEXT: */
// CHECK-NEXT: #define ATOMIC_ADD( x, v )  atomicAdd( &x, v );
#define ATOMIC_ADD( x, v )  atomicAdd( &x, v );

#define NUM_SYMBOLS 2

// CHECK:static void vlc_encode_kernel_sm64huff(uint8_t *dpct_local) {
// CHECK-NEXT:  unsigned int a = 1;
// CHECK-NEXT:  unsigned int kc = 1;
// CHECK-NEXT:  auto sm = (unsigned int *)dpct_local;
// CHECK-NEXT:  unsigned int* as			= (unsigned int*)(sm+2*NUM_SYMBOLS);
// CHECK-NEXT:  sycl::atomic<unsigned int, sycl::access::address_space::local_space>(sycl::local_ptr<unsigned int>(&as[kc])).fetch_or(a);
// CHECK-NEXT:}
__global__ static void vlc_encode_kernel_sm64huff() {
  unsigned int a = 1;
  unsigned int kc = 1;
  extern __shared__ unsigned int sm[];
  unsigned int* as			= (unsigned int*)(sm+2*NUM_SYMBOLS);
  atomicOr(&as[kc], a);
}

// CHECK:void addByte(unsigned int *s_WarpHist, unsigned int data) {
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1039:{{[0-9]+}}: The generated code assumes that "s_WarpHist + data" points to the global memory address space. If it points to a local memory address space, replace "sycl::global_ptr" with "sycl::local_ptr".
// CHECK-NEXT:  */
// CHECK-NEXT:  sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(s_WarpHist + data)).fetch_add(1);
// CHECK-NEXT:}
__device__ void addByte(unsigned int *s_WarpHist, unsigned int data) {
  atomicAdd(s_WarpHist + data, 1);
}

__global__ void histogram256Kernel() {
__shared__ unsigned int s_Hist[100];
  unsigned int *s_WarpHist= s_Hist + (threadIdx.x >> 1) * 10;
  addByte(s_WarpHist, 1000);
}

//CHECK:dpct::global_memory<volatile int, 0> g_mutex(0);
volatile __device__ int g_mutex = 0;

//CHECK:void __gpu_sync(int blocks_to_synch, volatile int *g_mutex) {
//CHECK-NEXT:  sycl::atomic<int>(sycl::global_ptr<int>((int *)g_mutex)).fetch_add(1);
//CHECK-NEXT:  while(*g_mutex < blocks_to_synch);
//CHECK-NEXT:}
__device__ void __gpu_sync(int blocks_to_synch) {
  atomicAdd((int *)&g_mutex, 1);
  while(g_mutex < blocks_to_synch);
}
