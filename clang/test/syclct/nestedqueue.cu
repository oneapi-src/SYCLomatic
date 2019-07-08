// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/nestedqueue.sycl.cpp --match-full-lines %s

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__host__ __device__ void foo1(){
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1021:{{[0-9]+}}: Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasHandle_t handle;
  cublasHandle_t handle;
  int n=1;
  float* x_S=0;
  int incx=1;
  int* result =0;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1020:{{[0-9]+}}: Migration of cublasIsamax_v2 if called from __global__ or __device__ function is not supported. You may need to redesign the code to use host-side mkl::isamax instead, which will submit this call to DPC++ queue automatically.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);
}

__device__ void foo2(){
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1021:{{[0-9]+}}: Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasHandle_t handle;
  cublasHandle_t handle;
  int n=1;
  float* x_S=0;
  int incx=1;
  int* result =0;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1020:{{[0-9]+}}: Migration of cublasIsamax_v2 if called from __global__ or __device__ function is not supported. You may need to redesign the code to use host-side mkl::isamax instead, which will submit this call to DPC++ queue automatically.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);
}

__global__ void foo3(){
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1021:{{[0-9]+}}: Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasHandle_t handle;
  cublasHandle_t handle;
  int n=1;
  float* x_S=0;
  int incx=1;
  int* result =0;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1020:{{[0-9]+}}: Migration of cublasIsamax_v2 if called from __global__ or __device__ function is not supported. You may need to redesign the code to use host-side mkl::isamax instead, which will submit this call to DPC++ queue automatically.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);
}

__host__ void foo4(){
  // CHECK: cl::sycl::queue handle;
  cublasHandle_t handle;
  int n=1;
  float* x_S=0;
  int incx=1;
  int* result =0;
  // CHECK: mkl::isamax(handle, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  cublasIsamax(handle, n, x_S, incx, result);
}