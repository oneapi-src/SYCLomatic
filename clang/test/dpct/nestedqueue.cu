// RUN: dpct --format-range=none --usm-level=none -out-root %T/nestedqueue %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/nestedqueue/nestedqueue.dp.cpp --match-full-lines %s

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__host__ __device__ void foo1(){
  // CHECK: /*
  // CHECK-NEXT: DPCT1021:{{[0-9]+}}: Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasHandle_t handle;
  cublasHandle_t handle;
  int n=1;
  float* x_S=0;
  int incx=1;
  int* result =0;
  // CHECK: /*
  // CHECK-NEXT: DPCT1020:{{[0-9]+}}: Migration of cublasIsamax, if it is called from __global__ or __device__ function, is not supported. You may need to redesign the code to use host-side oneapi::mkl::blas::column_major::iamax instead, which will submit this call to SYCL queue automatically.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);
}

__device__ void foo2(){
  // CHECK: /*
  // CHECK-NEXT: DPCT1021:{{[0-9]+}}: Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasHandle_t handle;
  cublasHandle_t handle;
  int n=1;
  float* x_S=0;
  int incx=1;
  int* result =0;
  // CHECK: /*
  // CHECK-NEXT: DPCT1020:{{[0-9]+}}: Migration of cublasIsamax, if it is called from __global__ or __device__ function, is not supported. You may need to redesign the code to use host-side oneapi::mkl::blas::column_major::iamax instead, which will submit this call to SYCL queue automatically.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);
}

__global__ void foo3(){
  // CHECK: /*
  // CHECK-NEXT: DPCT1021:{{[0-9]+}}: Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasHandle_t handle;
  cublasHandle_t handle;
  int n=1;
  float* x_S=0;
  int incx=1;
  int* result =0;
  // CHECK: /*
  // CHECK-NEXT: DPCT1020:{{[0-9]+}}: Migration of cublasIsamax, if it is called from __global__ or __device__ function, is not supported. You may need to redesign the code to use host-side oneapi::mkl::blas::column_major::iamax instead, which will submit this call to SYCL queue automatically.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);
}

__host__ void foo4(){
  // CHECK: dpct::queue_ptr handle;
  cublasHandle_t handle;
  int n=1;
  float* x_S=0;
  int incx=1;
  int* result =0;
  // CHECK: oneapi::mkl::blas::column_major::iamax(*handle, n, x_S_buf_ct1, incx, res_temp_buf_ct{{[0-9]+}});
  cublasIsamax(handle, n, x_S, incx, result);
}

