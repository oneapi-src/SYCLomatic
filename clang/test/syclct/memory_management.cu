// RUN: syclct -out-root %T %s -passes "MemoryTranslationRule" -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --match-full-lines --input-file %T/memory_management.sycl.cpp %s

#include <cuda_runtime.h>

void fooo() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  // CHECK: syclct::sycl_malloc((void **)&d_A, size);
  cudaMalloc((void **)&d_A, size);
  // CHECK: syclct::sycl_memset((void*)(d_A), (int)(0xf), (size_t)(size));
  cudaMemset(d_A, 0xf, size);
  // CHECK: syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_host);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK: syclct::sycl_free(d_A);
  cudaFree(d_A);
  free(h_A);
}

cudaError_t mallocWrapper(void **buffer, size_t size) {
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  return (syclct::sycl_malloc(buffer, size), 0);
  return cudaMalloc(buffer, size);
}

void checkError(cudaError_t err) {
}

void testCommas() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  float *d_B = NULL;
  // CHECK:  syclct::sycl_malloc((void **)&d_A, size);
  cudaMalloc((void **)&d_A, size);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  cudaError_t err = (syclct::sycl_malloc((void **)&d_A, size), 0);
  cudaError_t err = cudaMalloc((void **)&d_A, size);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((syclct::sycl_malloc((void **)&d_A, size), 0));
  checkError(cudaMalloc((void **)&d_A, size));
  // CHECK:  syclct::sycl_memset((void*)(d_A), (int)(0xf), (size_t)(size));
  cudaMemset(d_A, 0xf, size);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (syclct::sycl_memset((void*)(d_A), (int)(0xf), (size_t)(size)), 0);
  err = cudaMemset(d_A, 0xf, size);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((syclct::sycl_memset((void*)(d_A), (int)(0xf), (size_t)(size)), 0));
  checkError(cudaMemset(d_A, 0xf, size));

  ///////// Host to host
  // CHECK:  syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_host);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_host), 0);
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_host), 0));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost));

  ///////// Host to device
  // CHECK:  syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_device), 0);
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_device), 0));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

  ///////// Device to host
  // CHECK:  syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_host);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_host), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_host), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

  ///////// Device to Device
  // CHECK:  syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_device);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_device), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_device), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::automatic);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::automatic), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::automatic), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault));

  ///////// Host to device
  // CHECK:  syclct::sycl_memcpy_to_symbol(d_A.get_ptr(), (void*)(h_A), size, 0, syclct::host_to_device);
  cudaMemcpyToSymbol(d_A, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (syclct::sycl_memcpy_to_symbol(d_A.get_ptr(), (void*)(h_A), size, 0, syclct::host_to_device), 0);
  err = cudaMemcpyToSymbol(d_A, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((syclct::sycl_memcpy_to_symbol(d_A.get_ptr(), (void*)(h_A), size, 0, syclct::host_to_device), 0));
  checkError(cudaMemcpyToSymbol(d_A, h_A, size, 0, cudaMemcpyHostToDevice));

  ///////// Device to device
  // CHECK:  syclct::sycl_memcpy_to_symbol(d_B.get_ptr(), (void*)(d_B), size, 0, syclct::device_to_device);
  cudaMemcpyToSymbol(d_B, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (syclct::sycl_memcpy_to_symbol(d_B.get_ptr(), (void*)(d_B), size, 0, syclct::device_to_device), 0);
  err = cudaMemcpyToSymbol(d_B, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((syclct::sycl_memcpy_to_symbol(d_A.get_ptr(), (void*)(h_A), size, 0, syclct::device_to_device), 0));
  checkError(cudaMemcpyToSymbol(d_A, h_A, size, 0, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  syclct::sycl_memcpy_to_symbol(h_A.get_ptr(), (void*)(d_B), size, 0, syclct::automatic);
  cudaMemcpyToSymbol(h_A, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (syclct::sycl_memcpy_to_symbol(h_A.get_ptr(), (void*)(d_B), size, 0, syclct::automatic), 0);
  err = cudaMemcpyToSymbol(h_A, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((syclct::sycl_memcpy_to_symbol(h_A.get_ptr(), (void*)(d_B), size, 0, syclct::automatic), 0));
  checkError(cudaMemcpyToSymbol(h_A, d_B, size, 0, cudaMemcpyDefault));

  ///////// Default parameter overload
  // CHECK:  syclct::sycl_memcpy_to_symbol(h_A.get_ptr(), (void*)(d_B), size);
  cudaMemcpyToSymbol(h_A, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (syclct::sycl_memcpy_to_symbol(h_A.get_ptr(), (void*)(d_B), size), 0);
  err = cudaMemcpyToSymbol(h_A, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((syclct::sycl_memcpy_to_symbol(h_A.get_ptr(), (void*)(d_B), size), 0));
  checkError(cudaMemcpyToSymbol(h_A, d_B, size));

  ///////// Device to host
  // CHECK:  syclct::sycl_memcpy_from_symbol((void*)(d_A), h_A.get_ptr(), size, 0, syclct::device_to_host);
  cudaMemcpyFromSymbol(d_A, h_A, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (syclct::sycl_memcpy_from_symbol((void*)(d_A), h_A.get_ptr(), size, 0, syclct::device_to_host), 0);
  err = cudaMemcpyFromSymbol(d_A, h_A, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((syclct::sycl_memcpy_from_symbol((void*)(d_A), h_A.get_ptr(), size, 0, syclct::device_to_host), 0));
  checkError(cudaMemcpyFromSymbol(d_A, h_A, size, 0, cudaMemcpyDeviceToHost));

  ///////// Device to device
  // CHECK:  syclct::sycl_memcpy_from_symbol((void*)(d_B), d_B.get_ptr(), size, 0, syclct::device_to_device);
  cudaMemcpyFromSymbol(d_B, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (syclct::sycl_memcpy_from_symbol((void*)(d_B), d_B.get_ptr(), size, 0, syclct::device_to_device), 0);
  err = cudaMemcpyFromSymbol(d_B, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((syclct::sycl_memcpy_from_symbol((void*)(d_B), d_B.get_ptr(), size, 0, syclct::device_to_device), 0));
  checkError(cudaMemcpyFromSymbol(d_B, d_B, size, 0, cudaMemcpyDeviceToDevice));

  ///////// Default parameter overload
  // CHECK:  syclct::sycl_memcpy_from_symbol((void*)(h_A), d_B.get_ptr(), size);
  cudaMemcpyFromSymbol(h_A, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (syclct::sycl_memcpy_from_symbol((void*)(h_A), d_B.get_ptr(), size), 0);
  err = cudaMemcpyFromSymbol(h_A, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((syclct::sycl_memcpy_from_symbol((void*)(h_A), d_B.get_ptr(), size), 0));
  checkError(cudaMemcpyFromSymbol(h_A, d_B, size));

  // CHECK: syclct::sycl_free(d_A);
  cudaFree(d_A);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (syclct::sycl_free(d_A), 0);
  err = cudaFree(d_A);
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((syclct::sycl_free(d_A), 0));
  checkError(cudaFree(d_A));
  // CHECK:  free(h_A);
  free(h_A);
}

#define N 1024
void test_segmentation_fault() {
  float *buffer;
  /*
  * Original code in getSizeString():
  * "SizeExpr->getBeginLoc()" cannot get the real SourceLocation of "N*sizeof(float)",
  * and results in boundary violation in "SyclctGlobalInfo::getSourceManager().getCharacterData(SizeBegin)"
  * and fails with segmentation fault.
  * https://jira.devtools.intel.com/browse/CTST-527
  * https://jira.devtools.intel.com/browse/CTST-528
  */
  cudaMalloc(&buffer, N*sizeof(float));
}
