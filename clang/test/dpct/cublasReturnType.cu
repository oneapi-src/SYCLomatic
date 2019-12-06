// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasReturnType.dp.cpp --match-full-lines %s
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

// CHECK: int foo(int m, int n) {
cublasStatus_t foo(int m, int n) {
  // CHECK: return 0;
  return CUBLAS_STATUS_SUCCESS;
}

// CHECK: cl::sycl::queue foo1(int m) {
cublasHandle_t foo1(int m) {
  return 0;
}

// CHECK: cl::sycl::float2 foo2(cl::sycl::float2 m) {
cuComplex foo2(cuComplex m) {
  // CHECK: return cl::sycl::float2(1, 0);
  return make_cuComplex(1, 0);
}

// CHECK: cl::sycl::double2 foo3(cl::sycl::double2 m) {
cuDoubleComplex foo3(cuDoubleComplex m) {
  // CHECK: return cl::sycl::double2(1, 0);
  return make_cuDoubleComplex(1, 0);
}

// CHECK: mkl::transpose foo4(mkl::transpose m) {
cublasOperation_t foo4(cublasOperation_t m) {
  // CHECK: return mkl::transpose::conjtrans;
  return CUBLAS_OP_C;
}

// CHECK: mkl::uplo foo5(mkl::uplo m) {
cublasFillMode_t foo5(cublasFillMode_t m) {
  // CHECK: return mkl::uplo::lower;
  return CUBLAS_FILL_MODE_LOWER;
}

// CHECK: mkl::side foo6(mkl::side m) {
cublasSideMode_t foo6(cublasSideMode_t m) {
  // CHECK: return mkl::side::right;
  return CUBLAS_SIDE_RIGHT;
}

// CHECK: mkl::diag foo7(mkl::diag m) {
cublasDiagType_t foo7(cublasDiagType_t m) {
  // CHECK: return mkl::diag::nonunit;
  return CUBLAS_DIAG_NON_UNIT;
}
