// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/cublasReturnType.sycl.cpp --match-full-lines %s
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

// CHECK: std::complex<float> foo2(std::complex<float> m) {
cuComplex foo2(cuComplex m) {
  // CHECK: return std::complex<float>(1, 0);
  return make_cuComplex(1, 0);
}

// CHECK: std::complex<double> foo3(std::complex<double> m) {
cuDoubleComplex foo3(cuDoubleComplex m) {
  // CHECK: return std::complex<double>(1, 0);
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
