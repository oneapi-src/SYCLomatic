// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/memory_management.dp.cpp %s

#include <cuda_runtime.h>

__constant__ float constData[1234567 * 4];

void fooo() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  cudaStream_t stream;
  // CHECK: dpct::dpct_malloc((void **)&d_A, size);
  cudaMalloc((void **)&d_A, size);
  // CHECK: dpct::dpct_memset((void*)(d_A), 0xf, size);
  cudaMemset(d_A, 0xf, size);

  // CHECK: dpct::async_dpct_memset((void*)(d_A), 0xf, size);
  cudaMemsetAsync(d_A, 0xf, size);
  // CHECK: dpct::async_dpct_memset((void*)(d_A), 0xf, size);
  cudaMemsetAsync(d_A, 0xf, size, 0);
  // CHECK: dpct::async_dpct_memset((void*)(d_A), 0xf, size, stream);
  cudaMemsetAsync(d_A, 0xf, size, stream);

  // CHECK: dpct::dpct_memcpy((void*)(d_A), (void*)(h_A), size, dpct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: dpct::dpct_memcpy((void*)(h_A), (void*)(d_A), size, dpct::device_to_host);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

  // CHECK: dpct::async_dpct_memcpy((void*)(d_A), (void*)(h_A), size, dpct::host_to_device);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: dpct::async_dpct_memcpy((void*)(d_A), (void*)(h_A), size, dpct::host_to_device);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0);
  // CHECK: dpct::async_dpct_memcpy((void*)(d_A), (void*)(h_A), size, dpct::host_to_device, stream);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);

  // CHECK: dpct::async_dpct_memcpy((void*)(h_A), (void*)(d_A), size, dpct::device_to_host);
  cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy((void*)(h_A), (void*)(d_A), size, dpct::device_to_host);
  cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost, 0);
  // CHECK: dpct::async_dpct_memcpy((void*)(h_A), (void*)(d_A), size, dpct::device_to_host, stream);
  cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost, stream);

  // CHECK: dpct::async_dpct_memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: dpct::async_dpct_memcpy((void *)((char *)(constData.get_ptr()) + 2), (void*)(h_A), size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: dpct::async_dpct_memcpy((void *)((char *)(constData.get_ptr()) + 3), (void*)(h_A), size, dpct::host_to_device, stream);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);

  // CHECK: dpct::async_dpct_memcpy(constData.get_ptr(), (void*)(h_A), size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 0, cudaMemcpyHostToDevice);
  // dpct::async_dpct_memcpy(constData.get_ptr(), (void*)(h_A), size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 0, cudaMemcpyHostToDevice, 0);
  // dpct::async_dpct_memcpy(constData.get_ptr(), (void*)(h_A), size, dpct::host_to_device, stream);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 0, cudaMemcpyHostToDevice, stream);

  // CHECK: dpct::async_dpct_memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 2), size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: dpct::async_dpct_memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 3), size, dpct::device_to_host, stream);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);

  // CHECK: dpct::async_dpct_memcpy((void*)(h_A), constData.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 0, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy((void*)(h_A), constData.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 0, cudaMemcpyDeviceToHost, 0);
  // dpct::async_dpct_memcpy((void*)(h_A), constData.get_ptr(), size, dpct::device_to_host, stream);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 0, cudaMemcpyDeviceToHost, stream);

  // CHECK: dpct::dpct_free(d_A);
  cudaFree(d_A);
  free(h_A);
}

cudaError_t mallocWrapper(void **buffer, size_t size) {
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  return (dpct::dpct_malloc(buffer, size), 0);
  return cudaMalloc(buffer, size);
}

void checkError(cudaError_t err) {
}

void testCommas() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  float *d_B = NULL;
  // CHECK:  dpct::dpct_malloc((void **)&d_A, size);
  cudaMalloc((void **)&d_A, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  int err = (dpct::dpct_malloc((void **)&d_A, size), 0);
  cudaError_t err = cudaMalloc((void **)&d_A, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_malloc((void **)&d_A, size), 0));
  checkError(cudaMalloc((void **)&d_A, size));
  // CHECK:  dpct::dpct_memset((void*)(d_A), 0xf, size);
  cudaMemset(d_A, 0xf, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memset((void*)(d_A), 0xf, size), 0);
  err = cudaMemset(d_A, 0xf, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memset((void*)(d_A), 0xf, size), 0));
  checkError(cudaMemset(d_A, 0xf, size));

  ///////// Host to host
  // CHECK:  dpct::dpct_memcpy((void*)(d_A), (void*)(h_A), size, dpct::host_to_host);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void*)(d_A), (void*)(h_A), size, dpct::host_to_host), 0);
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((void*)(d_A), (void*)(h_A), size, dpct::host_to_host), 0));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost));

  ///////// Host to device
  // CHECK:  dpct::dpct_memcpy((void*)(d_A), (void*)(h_A), size, dpct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void*)(d_A), (void*)(h_A), size, dpct::host_to_device), 0);
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((void*)(d_A), (void*)(h_A), size, dpct::host_to_device), 0));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

  ///////// Device to host
  // CHECK:  dpct::dpct_memcpy((void*)(h_A), (void*)(d_A), size, dpct::device_to_host);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void*)(h_A), (void*)(d_A), size, dpct::device_to_host), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((void*)(h_A), (void*)(d_A), size, dpct::device_to_host), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

  ///////// Device to Device
  // CHECK:  dpct::dpct_memcpy((void*)(h_A), (void*)(d_A), size, dpct::device_to_device);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void*)(h_A), (void*)(d_A), size, dpct::device_to_device), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((void*)(h_A), (void*)(d_A), size, dpct::device_to_device), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  dpct::dpct_memcpy((void*)(h_A), (void*)(d_A), size, dpct::automatic);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void*)(h_A), (void*)(d_A), size, dpct::automatic), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((void*)(h_A), (void*)(d_A), size, dpct::automatic), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault));

  ///////// Host to device
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), (void*)(h_A), size, dpct::host_to_device);
  cudaMemcpyToSymbol(constData, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(constData.get_ptr(), (void*)(h_A), size, dpct::host_to_device), 0);
  err = cudaMemcpyToSymbol(constData, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(constData.get_ptr(), (void*)(h_A), size, dpct::host_to_device), 0));
  checkError(cudaMemcpyToSymbol(constData, h_A, size, 0, cudaMemcpyHostToDevice));

  // CHECK:  dpct::dpct_memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size, dpct::host_to_device);
  cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size, dpct::host_to_device), 0);
  err = cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size, dpct::host_to_device), 0));
  checkError(cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice));

  ///////// Device to device
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), (void*)(d_B), size, dpct::device_to_device);
  cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(constData.get_ptr(), (void*)(d_B), size, dpct::device_to_device), 0);
  err = cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(constData.get_ptr(), (void*)(h_A), size, dpct::device_to_device), 0));
  checkError(cudaMemcpyToSymbol(constData, h_A, size, 0, cudaMemcpyDeviceToDevice));

  // CHECK:  dpct::dpct_memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(d_B), size, dpct::device_to_device);
  cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(d_B), size, dpct::device_to_device), 0);
  err = cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size, dpct::device_to_device), 0));
  checkError(cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), (void*)(d_B), size, dpct::automatic);
  cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy(constData.get_ptr(), (void*)(d_B), size, dpct::automatic), 0);
  err = cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(constData.get_ptr(), (void*)(d_B), size, dpct::automatic), 0));
  checkError(cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDefault));

  // CHECK:  dpct::dpct_memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(d_B), size, dpct::automatic);
  cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(d_B), size, dpct::automatic), 0);
  err = cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(d_B), size, dpct::automatic), 0));
  checkError(cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDefault));

  ///////// Default parameter overload
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), (void*)(d_B), size);
  cudaMemcpyToSymbol(constData, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy(constData.get_ptr(), (void*)(d_B), size), 0);
  err = cudaMemcpyToSymbol(constData, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(constData.get_ptr(), (void*)(d_B), size), 0));
  checkError(cudaMemcpyToSymbol(constData, d_B, size));

  ///////// Device to host
  // CHECK:  dpct::dpct_memcpy((void*)(h_A), constData.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbol(h_A, constData, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void*)(h_A), constData.get_ptr(), size, dpct::device_to_host), 0);
  err = cudaMemcpyFromSymbol(h_A, constData, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((void*)(h_A), constData.get_ptr(), size, dpct::device_to_host), 0));
  checkError(cudaMemcpyFromSymbol(h_A, constData, size, 0, cudaMemcpyDeviceToHost));

  // CHECK:  dpct::dpct_memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size, dpct::device_to_host);
  cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size, dpct::device_to_host), 0);
  err = cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size, dpct::device_to_host), 0));
  checkError(cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost));

  ///////// Device to device
  // CHECK:  dpct::dpct_memcpy((void*)(d_B), constData.get_ptr(), size, dpct::device_to_device);
  cudaMemcpyFromSymbol(d_B, constData, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void*)(d_B), constData.get_ptr(), size, dpct::device_to_device), 0);
  err = cudaMemcpyFromSymbol(d_B, constData, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy((void*)(d_B), constData.get_ptr(), size, dpct::device_to_device), 0));
  checkError(cudaMemcpyFromSymbol(d_B, constData, size, 0, cudaMemcpyDeviceToDevice));


  // CHECK:  dpct::dpct_memcpy((void*)(d_B), (void *)((char *)(constData.get_ptr()) + 1), size, dpct::device_to_device);
  cudaMemcpyFromSymbol(d_B, constData, size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void*)(d_B), (void *)((char *)(constData.get_ptr()) + 1), size, dpct::device_to_device), 0);
  err = cudaMemcpyFromSymbol(d_B, constData, size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy((void*)(d_B), (void *)((char *)(constData.get_ptr()) + 1), size, dpct::device_to_device), 0));
  checkError(cudaMemcpyFromSymbol(d_B, constData, size, 1, cudaMemcpyDeviceToDevice));

  ///////// Default parameter overload
  // CHECK:  dpct::dpct_memcpy((void*)(h_A), constData.get_ptr(), size);
  cudaMemcpyFromSymbol(h_A, constData, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy((void*)(h_A), constData.get_ptr(), size), 0);
  err = cudaMemcpyFromSymbol(h_A, constData, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy((void*)(h_A), constData.get_ptr(), size), 0));
  checkError(cudaMemcpyFromSymbol(h_A, constData, size));

  // CHECK: dpct::dpct_free(d_A);
  cudaFree(d_A);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_free(d_A), 0);
  err = cudaFree(d_A);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_free(d_A), 0));
  checkError(cudaFree(d_A));
  // CHECK:  free(h_A);
  free(h_A);
}

// CHECK:  dpct::device_memory<float, 1> d_A(1234567);
// CHECK:  dpct::device_memory<float, 1> d_B(1234567);
static __device__ float d_A[1234567];
static __device__ float d_B[1234567];

void testCommas_in_device_memory() {
  size_t size = 1234567 * sizeof(float);
  cudaError_t err;
  float *h_A = (float *)malloc(size);

  // CHECK:  dpct::dpct_memset(d_A.get_ptr(), 0xf, size);
  cudaMemset(d_A, 0xf, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memset(d_A.get_ptr(), 0xf, size), 0);
  err = cudaMemset(d_A, 0xf, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memset(d_A.get_ptr(), 0xf, size), 0));
  checkError(cudaMemset(d_A, 0xf, size));

  ///////// Host to host
  // CHECK:  dpct::dpct_memcpy((void*)(h_A), (void*)(h_A), size, dpct::host_to_host);
  cudaMemcpy(h_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void*)(h_A), (void*)(h_A), size, dpct::host_to_host), 0);
  err = cudaMemcpy(h_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((void*)(h_A), (void*)(h_A), size, dpct::host_to_host), 0));
  checkError(cudaMemcpy(h_A, h_A, size, cudaMemcpyHostToHost));

  ///////// Host to device
  // CHECK:  dpct::dpct_memcpy(d_A.get_ptr(), (void*)(h_A), size, dpct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_A.get_ptr(), (void*)(h_A), size, dpct::host_to_device), 0);
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(d_A.get_ptr(), (void*)(h_A), size, dpct::host_to_device), 0));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

  ///////// Device to host
  // CHECK:  dpct::dpct_memcpy((void*)(h_A), d_A.get_ptr(), size, dpct::device_to_host);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void*)(h_A), d_A.get_ptr(), size, dpct::device_to_host), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((void*)(h_A), d_A.get_ptr(), size, dpct::device_to_host), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

  ///////// Device to Device
  // CHECK:  dpct::dpct_memcpy(d_B.get_ptr(), d_A.get_ptr(), size, dpct::device_to_device);
  cudaMemcpy(d_B, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_B.get_ptr(), d_A.get_ptr(), size, dpct::device_to_device), 0);
  err = cudaMemcpy(d_B, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(d_B.get_ptr(), d_A.get_ptr(), size, dpct::device_to_device), 0));
  checkError(cudaMemcpy(d_B, d_A, size, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  dpct::dpct_memcpy((void*)(h_A), d_A.get_ptr(), size, dpct::automatic);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void*)(h_A), d_A.get_ptr(), size, dpct::automatic), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((void*)(h_A), d_A.get_ptr(), size, dpct::automatic), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault));

  ///////// Host to device
  // CHECK:  dpct::dpct_memcpy(d_A.get_ptr(), (void*)(h_A), size, dpct::host_to_device);
  cudaMemcpyToSymbol(d_A, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_A.get_ptr(), (void*)(h_A), size, dpct::host_to_device), 0);
  err = cudaMemcpyToSymbol(d_A, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(d_A.get_ptr(), (void*)(h_A), size, dpct::host_to_device), 0));
  checkError(cudaMemcpyToSymbol(d_A, h_A, size, 0, cudaMemcpyHostToDevice));

  ///////// Device to device
  // CHECK:  dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device);
  cudaMemcpyToSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  err = cudaMemcpyToSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device), 0));
  checkError(cudaMemcpyToSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  dpct::dpct_memcpy((void*)(h_A), d_B.get_ptr(), size, dpct::automatic);
  cudaMemcpyToSymbol(h_A, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy((void*)(h_A), d_B.get_ptr(), size, dpct::automatic), 0);
  err = cudaMemcpyToSymbol(h_A, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy((void*)(h_A), d_B.get_ptr(), size, dpct::automatic), 0));
  checkError(cudaMemcpyToSymbol(h_A, d_B, size, 0, cudaMemcpyDefault));

  ///////// Default parameter overload
  // CHECK:  dpct::dpct_memcpy((void*)(h_A), d_B.get_ptr(), size);
  cudaMemcpyToSymbol(h_A, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy((void*)(h_A), d_B.get_ptr(), size), 0);
  err = cudaMemcpyToSymbol(h_A, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy((void*)(h_A), d_B.get_ptr(), size), 0));
  checkError(cudaMemcpyToSymbol(h_A, d_B, size));

  ///////// Device to host
  // CHECK:  dpct::dpct_memcpy((void*)(h_A), d_A.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbol(h_A, d_A, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((void*)(h_A), d_A.get_ptr(), size, dpct::device_to_host), 0);
  err = cudaMemcpyFromSymbol(h_A, d_A, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((void*)(h_A), d_A.get_ptr(), size, dpct::device_to_host), 0));
  checkError(cudaMemcpyFromSymbol(h_A, d_A, size, 0, cudaMemcpyDeviceToHost));

  ///////// Device to device
  // CHECK:  dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device);
  cudaMemcpyFromSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device), 0);
  err = cudaMemcpyFromSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device), 0));
  checkError(cudaMemcpyFromSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice));

  ///////// Default parameter overload
  // CHECK:  dpct::dpct_memcpy((void*)(h_A), d_B.get_ptr(), size);
  cudaMemcpyFromSymbol(h_A, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy((void*)(h_A), d_B.get_ptr(), size), 0);
  err = cudaMemcpyFromSymbol(h_A, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy((void*)(h_A), d_B.get_ptr(), size), 0));
  checkError(cudaMemcpyFromSymbol(h_A, d_B, size));

  void *p_addr;
  // CHECK:  *(&p_addr) = d_A.get_ptr();
  cudaGetSymbolAddress(&p_addr, d_A);

  // CHECK:  /*
  // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  err = (*(&p_addr) = d_A.get_ptr(), 0);
  err = cudaGetSymbolAddress(&p_addr, d_A);

  // CHECK: /*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:checkError((*(&p_addr) = d_A.get_ptr(), 0));
  checkError(cudaGetSymbolAddress(&p_addr, d_A));

  size_t size2;
  // CHECK: size2 = d_A.get_size();
  cudaGetSymbolSize(&size2, d_A);

  // CHECK: /*
  // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  err = (size2 = d_A.get_size(), 0);
  err = cudaGetSymbolSize(&size2, d_A);

  // CHECK: /*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:checkError((size2 = d_A.get_size(), 0));
  checkError(cudaGetSymbolSize(&size2, d_A));

  int* a;
  cudaStream_t stream;

  err = cudaMallocManaged(&a, 100);
  // CHECK: stream?(stream)->prefetch(a,100):dpct::get_device_manager().get_device(0).default_queue().prefetch(a,100);
  cudaMemPrefetchAsync (a, 100, 0, stream);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: err = (stream?(stream)->prefetch(a,100):dpct::get_device_manager().get_device(0).default_queue().prefetch(a,100), 0);
  err = cudaMemPrefetchAsync (a, 100, 0, stream);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkError((stream?(stream)->prefetch(a,100):dpct::get_device_manager().get_device(0).default_queue().prefetch(a,100), 0));
  checkError(cudaMemPrefetchAsync (a, 100, 0, stream));

  // CHECK:  free(h_A);
  free(h_A);
}

#define CUDA_CHECK(call)                                                           \
    if ((call) != cudaSuccess) { \
        exit(-1); \
    }

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {}


template<typename T>
void uninstantiated_template_call(const T * d_data, size_t width, size_t height) {
  size_t datasize = width * height;
  T * data = new T[datasize];
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  assert_cuda((dpct::dpct_memcpy((void*)(data), (void*)(d_data), datasize * sizeof(T), dpct::device_to_host), 0));
  assert_cuda(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

  // CHECK: dpct::dpct_memcpy((void*)(data), (void*)(d_data), datasize * sizeof(T), dpct::device_to_host);
  cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

#define DATAMACRO data+32*32

  // CHECK: dpct::dpct_memcpy((void*)(DATAMACRO), (void*)(d_data), datasize * sizeof(T), dpct::device_to_host);
  cudaMemcpy(DATAMACRO, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

  // CHECK: dpct::dpct_memcpy((void*)(32*32+DATAMACRO), (void*)(d_data), datasize * sizeof(T), dpct::device_to_host);
  cudaMemcpy(32*32+DATAMACRO, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((void*)(data), (void*)(d_data), datasize * sizeof(T), dpct::device_to_host), 0));
  checkError(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT: int err = (dpct::dpct_memcpy((void*)(data), (void*)(d_data), datasize * sizeof(T), dpct::device_to_host), 0);
  cudaError_t err = cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT: CUDA_CHECK((dpct::dpct_memcpy((void*)(data), (void*)(d_data), datasize * sizeof(T), dpct::device_to_host), 0));
  CUDA_CHECK(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT: checkCudaErrors((dpct::dpct_memcpy((void*)(data), (void*)(d_data), datasize * sizeof(T), dpct::device_to_host), 0));
  checkCudaErrors(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

  // CHECK: #define CUDAMEMCPY dpct::dpct_memcpy
  // CHECK-NEXT: CUDAMEMCPY((void*)(data), (void*)(d_data), datasize * sizeof(T), dpct::device_to_host);
  #define CUDAMEMCPY cudaMemcpy
  CUDAMEMCPY(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

  delete[] data;
}

#define N 1024
void test_segmentation_fault() {
  float *buffer;
  /*
  * Original code in getSizeString():
  * "SizeExpr->getBeginLoc()" cannot get the real SourceLocation of "N*sizeof(float)",
  * and results in boundary violation in "dpctGlobalInfo::getSourceManager().getCharacterData(SizeBegin)"
  * and fails with segmentation fault.
  * https://jira.devtools.intel.com/browse/CTST-527
  * https://jira.devtools.intel.com/browse/CTST-528
  */
  cudaMalloc(&buffer, N*sizeof(float));
}

// CHECK: dpct::device_memory<uint32_t, 1> d_error(1);
static __device__ uint32_t d_error[1];

void test_foo(){
  // CHECK: dpct::dpct_memset(d_error.get_ptr(), 0, sizeof(uint32_t));
  cudaMemset(d_error, 0, sizeof(uint32_t));
}
