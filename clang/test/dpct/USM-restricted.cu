// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --usm-level=restricted -out-root %T %s -- -std=c++14 -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --match-full-lines --input-file %T/USM-restricted.dp.cpp %s

#include <cuda_runtime.h>

void foo() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;

  // CHECK: dpct::dpct_malloc((void **)&d_A, size);
  cudaMalloc((void **)&d_A, size);

  // CHECK: *((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context());
  cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault);

  // CHECK: *((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context());
  cudaMallocHost((void **)&h_A, size);

  // CHECK: *((void **)&d_A) = cl::sycl::malloc_shared(size, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
  cudaMallocManaged((void **)&d_A, size);

  // CHECK: cl::sycl::free(h_A, dpct::get_default_queue().get_context());
  cudaFreeHost(h_A);

  // CHECK: *(&d_A) = h_A;
  cudaHostGetDevicePointer(&d_A, h_A, 0);

  cudaHostRegister(h_A, size, 0);
  cudaHostUnregister(h_A);
}
