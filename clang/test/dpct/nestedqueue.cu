// UNSUPPORTED: cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-11.9, cuda-12.0, cuda-12.1
// UNSUPPORTED: v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v11.9, v12.0, v12.1
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
  // CHECK-NEXT: DPCT1020:{{[0-9]+}}: Migration of cublasIsamax, if it is called from __global__ or __device__ function, is not supported. You may need to redesign the code to use the host-side oneapi::mkl::blas::column_major::iamax instead, which submits this call to the SYCL queue automatically.
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
  // CHECK-NEXT: DPCT1020:{{[0-9]+}}: Migration of cublasIsamax, if it is called from __global__ or __device__ function, is not supported. You may need to redesign the code to use the host-side oneapi::mkl::blas::column_major::iamax instead, which submits this call to the SYCL queue automatically.
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
  // CHECK-NEXT: DPCT1020:{{[0-9]+}}: Migration of cublasIsamax, if it is called from __global__ or __device__ function, is not supported. You may need to redesign the code to use the host-side oneapi::mkl::blas::column_major::iamax instead, which submits this call to the SYCL queue automatically.
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

