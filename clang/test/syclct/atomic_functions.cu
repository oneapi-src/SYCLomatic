// RUN: syclct -out-root %T %s -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/atomic_functions.sycl.cpp --match-full-lines %s

#include <cuda_runtime.h>

#include <iostream>
#include <memory>

#define NUM_ATOMICS 11

template <typename T>
__global__ void test(T *data) {
  // CHECK: T tid = item_{{[a-f0-9]+}}.get_local_id(0);
  T tid = threadIdx.x;

  // CHECK: syclct::atomic_fetch_add(&data[0], (T)(tid));
  atomicAdd(&data[0], tid);

  // CHECK: syclct::atomic_fetch_sub(&data[1], (T)(tid));
  atomicSub(&data[1], tid);

  // CHECK: syclct::atomic_exchange(&data[2], (T)(tid));
  atomicExch(&data[2], tid);

  // CHECK: syclct::atomic_fetch_max(&data[3], (T)(tid));
  atomicMax(&data[3], tid);

  // CHECK: syclct::atomic_fetch_min(&data[4], (T)(tid));
  atomicMin(&data[4], tid);

  // CHECK: /*
  // CHECK: SYCLCT1007:0: atomicInc: not support API, need manual porting.
  // CHECK: */
  atomicInc((unsigned int *)&data[5], (unsigned int)tid);

  // CHECK: /*
  // CHECK: SYCLCT1007:1: atomicDec: not support API, need manual porting.
  // CHECK: */
  atomicDec((unsigned int *)&data[6], (unsigned int)tid);

  // CHECK: syclct::atomic_compare_exchange_strong(&data[7], (T)(tid - 1), (T)(tid));
  atomicCAS(&data[7], tid - 1, tid);

  // CHECK: syclct::atomic_fetch_and(&data[8], (T)(tid));
  atomicAnd(&data[8], tid);

  // CHECK: syclct::atomic_fetch_or(&data[9], (T)(tid));
  atomicOr(&data[9], tid);

  // CHECK: syclct::atomic_fetch_xor(&data[10], (T)(tid));
  atomicXor(&data[10], tid);
}

template <>
__global__ void test(unsigned long long int* data) {
  unsigned long long int tid = threadIdx.x;

  // CHECK: syclct::atomic_fetch_add(&data[0], (unsigned long long)(tid));
  atomicAdd(&data[0], tid);

  // CHECK: syclct::atomic_exchange(&data[2], (unsigned long long)(tid));
  atomicExch(&data[2], tid);

  // CHECK: syclct::atomic_compare_exchange_strong(&data[7], (unsigned long long)(tid - 1), (unsigned long long)(tid));
  atomicCAS(&data[7], tid - 1, tid);
}

template <>
__global__ void test(float* data) {
  float tid = threadIdx.x;

  // CHECK: syclct::atomic_fetch_add(&data[0], (float)(tid));
  atomicAdd(&data[0], tid);

  // CHECK: syclct::atomic_exchange(&data[2], (float)(tid));
  atomicExch(&data[2], tid);
}

template <>
__global__ void test(double* data) {
  double tid = threadIdx.x;

  // CHECK: syclct::atomic_fetch_add(&data[0], (double)(tid));
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
  // CHECK:   std::pair<syclct::buffer_t, size_t> dev_ptr_buf = syclct::get_buffer_and_offset(dev_ptr);
  // CHECK:   size_t dev_ptr_offset = dev_ptr_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto dev_ptr_acc = dev_ptr_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class test_{{[a-f0-9]+}}, T>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(k_threads_per_block, 1, 1)), cl::sycl::range<3>(k_threads_per_block, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           T *dev_ptr = (T*)(&dev_ptr_acc[0] + dev_ptr_offset);
  // CHECK:           test<T>(dev_ptr, [[ITEM]]);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  test<T><<<1, k_threads_per_block>>>(dev_ptr);
}

int main() {
  InvokeKernel<int>();
  InvokeKernel<unsigned int>();
  InvokeKernel<unsigned long long int>();
  InvokeKernel<float>();
  InvokeKernel<double>();
}
