// RUN: syclct -out-root %T %s -passes "MemoryTranslationRule" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/memory_management.sycl.cpp %s

#include <cuda_runtime.h>

void fooo() {
    size_t size = 1234567 * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *d_A = NULL;
    // CHECK: syclct::sycl_malloc<char>((void **)&d_A, size);
    cudaMalloc((void **)&d_A, size);
    // CHECK: syclct::sycl_memset((void*)(d_A), (void*)(15), (void*)(size));
    cudaMemset(d_A, 0xf, size);
    // CHECK: syclct::sycl_memcpy<char>(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    // CHECK: syclct::sycl_memcpy<char>(h_A, d_A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    // CHECK: syclct::sycl_free<char>(d_A);
    cudaFree(d_A);
    free(h_A);
}

cudaError_t mallocWrapper(void **buffer, size_t size) {
  // CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  return (syclct::sycl_malloc<char>(buffer, size), 0);
  return cudaMalloc(buffer, size);
}

void checkError(cudaError_t err) {

}

int testCommas() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
// CHECK:  syclct::sycl_malloc<char>((void **)&d_A, size);
  cudaMalloc((void **)&d_A, size);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  cudaError_t err = (syclct::sycl_malloc<char>((void **)&d_A, size), 0);
  cudaError_t err = cudaMalloc((void **)&d_A, size);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  checkError((syclct::sycl_malloc<char>((void **)&d_A, size), 0));
  checkError(cudaMalloc((void **)&d_A, size));
// CHECK:  syclct::sycl_memset((void*)(d_A), (void*)(15), (void*)(size));
  cudaMemset(d_A, 0xf, size);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  err = (syclct::sycl_memset((void*)(d_A), (void*)(15), (void*)(size)), 0);
  err = cudaMemset(d_A, 0xf, size);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  checkError((syclct::sycl_memset((void*)(d_A), (void*)(15), (void*)(size)), 0));
  checkError(cudaMemset(d_A, 0xf, size));
// CHECK:  syclct::sycl_memcpy<char>(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  err = (syclct::sycl_memcpy<char>(d_A, h_A, size, cudaMemcpyHostToDevice), 0);
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  checkError((syclct::sycl_memcpy<char>(d_A, h_A, size, cudaMemcpyHostToDevice), 0));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
// CHECK:  syclct::sycl_memcpy<char>(h_A, d_A, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  err = (syclct::sycl_memcpy<char>(h_A, d_A, size, cudaMemcpyDeviceToHost), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  checkError((syclct::sycl_memcpy<char>(h_A, d_A, size, cudaMemcpyDeviceToHost), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));
// CHECK:  syclct::sycl_free<char>(d_A);
  cudaFree(d_A);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  err = (syclct::sycl_free<char>(d_A), 0);
  err = cudaFree(d_A);
// CHECK:/*
// CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:  checkError((syclct::sycl_free<char>(d_A), 0));
  checkError(cudaFree(d_A));
// CHECK:  free(h_A);
  free(h_A);
}