// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --usm-level=restricted -out-root %T %s -- -std=c++14 -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --match-full-lines --input-file %T/USM-restricted.dp.cpp %s

#include <cuda_runtime.h>

__constant__ float constData[1234567 * 4];

void foo() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;

  cudaStream_t stream;

  // CHECK: *((void **)&d_A) = cl::sycl::malloc_device(size, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
  cudaMalloc((void **)&d_A, size);

  /// memcpy
  // CHECK: dpct::get_default_queue().memcpy((void*)(d_A), (void*)(h_A), size).wait();
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  /// memcpy async
  // CHECK: dpct::get_default_queue().memcpy((void*)(d_A), (void*)(h_A), size);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: dpct::get_default_queue().memcpy((void*)(d_A), (void*)(h_A), size);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0);
  // CHECK: stream.memcpy((void*)(d_A), (void*)(h_A), size);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);

  /// memcpy from symbol
  // CHECK: dpct::get_default_queue().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size).wait();
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: dpct::get_default_queue().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size).wait();
  cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);

  /// memcpy from symbol async
  // CHECK: dpct::get_default_queue().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: dpct::get_default_queue().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 2), size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: stream.memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 3), size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);

  /// memcpy to symbol
  // CHECK: dpct::get_default_queue().memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size).wait();
  cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: dpct::get_default_queue().memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size).wait();
  cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);

  /// memcpy to symbol async
  // CHECK: dpct::get_default_queue().memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: dpct::get_default_queue().memcpy((void *)((char *)(constData.get_ptr()) + 2), (void*)(h_A), size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: stream.memcpy((void *)((char *)(constData.get_ptr()) + 3), (void*)(h_A), size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);

  /// memset
  // CHECK: dpct::get_default_queue().memset((void*)(d_A), 23, size).wait();
  cudaMemset(d_A, 23, size);

  /// memset async
  // CHECK: dpct::get_default_queue().memset((void*)(d_A), 23, size);
  cudaMemsetAsync(d_A, 23, size);
  // CHECK: dpct::get_default_queue().memset((void*)(d_A), 23, size);
  cudaMemsetAsync(d_A, 23, size, 0);
  // CHECK: stream.memset((void*)(d_A), 23, size);
  cudaMemsetAsync(d_A, 23, size, stream);

  // CHECK: *((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context());
  cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault);

  // CHECK: *((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context());
  cudaMallocHost((void **)&h_A, size);
  // CHECK: *((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context());
  cudaMallocHost(&h_A, size);

  // CHECK: *((void **)&d_A) = cl::sycl::malloc_shared(size, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
  cudaMallocManaged((void **)&d_A, size);

  // CHECK: cl::sycl::free(h_A, dpct::get_default_queue().get_context());
  cudaFreeHost(h_A);

  // CHECK: *(&d_A) = h_A;
  cudaHostGetDevicePointer(&d_A, h_A, 0);

  cudaHostRegister(h_A, size, 0);
  cudaHostUnregister(h_A);
}
