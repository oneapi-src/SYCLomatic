// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/atomic_functions.dp.cpp --match-full-lines %s

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

  // CHECK: /*
  // CHECK: DPCT1007:0: Migration of this CUDA API is not supported by the Intel(R) DPC++ Compatibility Tool.
  // CHECK: */
  atomicInc((unsigned int *)&data[5], (unsigned int)tid);

  // CHECK: /*
  // CHECK: DPCT1007:1: Migration of this CUDA API is not supported by the Intel(R) DPC++ Compatibility Tool.
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

  // CHECK: dpct::atomic_fetch_add(&data[0], tid);
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
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> dev_ptr_buf_ct0 = dpct::get_buffer_and_offset(dev_ptr);
  // CHECK-NEXT:   size_t dev_ptr_offset_ct0 = dev_ptr_buf_ct0.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto dev_ptr_acc_ct0 = dev_ptr_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class test_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, k_threads_per_block), cl::sycl::range<3>(1, 1, k_threads_per_block)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           T *dev_ptr_ct0 = (T *)(&dev_ptr_acc_ct0[0] + dev_ptr_offset_ct0);
  // CHECK-NEXT:           test<T>(dev_ptr_ct0, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  test<T><<<1, k_threads_per_block>>>(dev_ptr);
}

// CHECK: dpct::device_memory<uint32_t, 1> d_error(1);
static __device__ uint32_t d_error[1];

// CHECK: void fun(dpct::accessor<uint32_t, dpct::device, 1> d_error){
__device__ void fun(){
  double *a;
  float b;
  // CHECK: dpct::atomic_fetch_add(a, (double)(1));
  atomicAdd(a, 1);

  // CHECK: dpct::atomic_fetch_add(a, (double)(b));
  atomicAdd(a, b);

  // CHECK: dpct::atomic_fetch_add((uint32_t*)(d_error), (uint32_t)(1));
  atomicAdd(d_error, 1);
}

int main() {
  InvokeKernel<int>();
  InvokeKernel<unsigned int>();
  InvokeKernel<unsigned long long int>();
  InvokeKernel<float>();
  InvokeKernel<double>();
}
