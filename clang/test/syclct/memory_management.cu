// RUN: syclct -out-root %T %s -passes "MemoryTranslationRule" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/memory_management.sycl.cpp %s

#include <cuda_runtime.h>

void fooo() {
    size_t size = 1234567 * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *d_A = NULL;
    // CHECK: syclct::sycl_malloc((void **)&d_A, size);
    cudaMalloc((void **)&d_A, size);
    // CHECK: syclct::sycl_memset((void*)(d_A), (int)(15), (size_t)(size));
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
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
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
// CHECK:  syclct::sycl_malloc((void **)&d_A, size);
  cudaMalloc((void **)&d_A, size);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  cudaError_t err = (syclct::sycl_malloc((void **)&d_A, size), 0);
  cudaError_t err = cudaMalloc((void **)&d_A, size);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  checkError((syclct::sycl_malloc((void **)&d_A, size), 0));
  checkError(cudaMalloc((void **)&d_A, size));
// CHECK:  syclct::sycl_memset((void*)(d_A), (int)(15), (size_t)(size));
  cudaMemset(d_A, 0xf, size);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  err = (syclct::sycl_memset((void*)(d_A), (int)(15), (size_t)(size)), 0);
  err = cudaMemset(d_A, 0xf, size);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  checkError((syclct::sycl_memset((void*)(d_A), (int)(15), (size_t)(size)), 0));
  checkError(cudaMemset(d_A, 0xf, size));

///////// Host to host
// CHECK:  syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_host);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  err = (syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_host), 0);
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  checkError((syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_host), 0));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost));

///////// Host to device
// CHECK:  syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  err = (syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_device), 0);
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  checkError((syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_device), 0));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

///////// Device to host
// CHECK:  syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_host);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  err = (syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_host), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  checkError((syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_host), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

///////// Device to Device
// CHECK:  syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_device);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  err = (syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_device), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  checkError((syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_device), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice));

///////// Default
// CHECK:  syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::automatic);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  err = (syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::automatic), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  checkError((syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::automatic), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault));

// CHECK:  syclct::sycl_free(d_A);
  cudaFree(d_A);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  err = (syclct::sycl_free(d_A), 0);
  err = cudaFree(d_A);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  checkError((syclct::sycl_free(d_A), 0));
  checkError(cudaFree(d_A));
// CHECK:  free(h_A);
  free(h_A);
}
