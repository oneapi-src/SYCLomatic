// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/USM-none.dp.cpp %s

// CHECK: #define DPCT_USM_LEVEL_NONE
// CHECK-NEXT: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

__constant__ float constData[1234567 * 4];

void foo() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  cudaStream_t stream;

  // CHECK: dpct::dpct_malloc((void **)&d_A, size);
  cudaMalloc((void **)&d_A, size);

  /// memcpy
  // CHECK: dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  /// memcpy async
  // CHECK: dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0);
  // CHECK: dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device, stream);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);

  /// memcpy from symbol
  // CHECK: dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host);
  cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);

  /// memcpy from symbol async
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 2, size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 3, size, dpct::device_to_host, stream);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);

  /// memcpy to symbol
  // CHECK: dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);

  /// memcpy to symbol async
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 2, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 3, h_A, size, dpct::host_to_device, stream);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);

  /// memset
  // CHECK: dpct::dpct_memset(d_A, 23, size);
  cudaMemset(d_A, 23, size);

  /// memset async
  // CHECK: dpct::async_dpct_memset(d_A, 23, size);
  cudaMemsetAsync(d_A, 23, size);
  // CHECK: dpct::async_dpct_memset(d_A, 23, size);
  cudaMemsetAsync(d_A, 23, size, 0);
  // CHECK: dpct::async_dpct_memset(d_A, 23, size, stream);
  cudaMemsetAsync(d_A, 23, size, stream);

  // CHECK: h_A = (float *)malloc(size);
  cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault);
  // CHECK: h_A = (float *)malloc(size);
  cudaMallocHost((void **)&h_A, size);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  cudaMallocManaged((void **)&d_A, size);

  // CHECK: free(h_A);
  cudaFreeHost(h_A);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  cudaHostGetDevicePointer(&d_A, h_A, 0);

  cudaHostRegister(h_A, size, 0);
  cudaHostUnregister(h_A);
}
