// RUN: cat %s > %T/cublas-lambda.cu
// RUN: cd %T
// RUN: dpct --usm-level=none -out-root %T/cublas-lambda cublas-lambda.cu --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-lambda/cublas-lambda.dp.cpp --match-full-lines cublas-lambda.cu
// RUN: %if build_lit %{icpx -c -fsycl %T/cublas-lambda/cublas-lambda.dp.cpp -o %T/cublas-lambda/cublas-lambda.dp.o %}

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <cstdio>
// CHECK: #include <dpct/blas_utils.hpp>
#include <cstdio>
#include "cublas_v2.h"
#include <cuda_runtime.h>

cublasStatus_t status;
cublasHandle_t handle;
int N = 275;
float *d_A_H = 0;
float *d_B_H = 0;
float *d_C_H = 0;
float alpha_H = 1.0f;
float beta_H = 0.0f;

int main() {
  // CHECK: if (DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:         handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:         dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_H)), N,
  // CHECK-NEXT:         dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_H)), N,
  // CHECK-NEXT:         beta_H,
  // CHECK-NEXT:         dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_H)), N))) {
  // CHECK-NEXT: } else if (DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:                handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:                dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_H)),
  // CHECK-NEXT:                N,
  // CHECK-NEXT:                dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_H)),
  // CHECK-NEXT:                N, beta_H,
  // CHECK-NEXT:                dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_H)),
  // CHECK-NEXT:                N))) {
  // CHECK-NEXT: }
  if (cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N)) {
  }
  else if (cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N)) {
  }


  // CHECK: if (int stat = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:         handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:         dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_H)), N,
  // CHECK-NEXT:         dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_H)), N,
  // CHECK-NEXT:         beta_H,
  // CHECK-NEXT:         dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_H)), N))) {
  // CHECK-NEXT: }
  if(int stat = cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N)){
  }


  // CHECK: for (oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:          handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:          dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_H)), N,
  // CHECK-NEXT:          dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_H)), N,
  // CHECK-NEXT:          beta_H,
  // CHECK-NEXT:          dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_H)), N);
  // CHECK-NEXT:      ;) {
  // CHECK-NEXT: }
  for(cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);;){
  }

  // CHECK: for (; DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:          handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:          dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_H)), N,
  // CHECK-NEXT:          dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_H)), N,
  // CHECK-NEXT:          beta_H,
  // CHECK-NEXT:          dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_H)),
  // CHECK-NEXT:          N));) {
  // CHECK-NEXT: }
  for(;cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);){
  }

  // CHECK: while (DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:            handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:            dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_H)), N,
  // CHECK-NEXT:            dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_H)), N,
  // CHECK-NEXT:            beta_H,
  // CHECK-NEXT:            dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_H)),
  // CHECK-NEXT:            N)) != 0) {
  // CHECK-NEXT: }
  while(cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N)!=0){
  }



  // CHECK: do{
  // CHECK-NEXT: } while (DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:     handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:     dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_H)), N,
  // CHECK-NEXT:     dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_H)), N, beta_H,
  // CHECK-NEXT:     dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_H)), N)));
  do{
  }while(cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N));


  // CHECK: switch (
  // CHECK-NEXT:     int stat = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:         handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:         dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_H)), N,
  // CHECK-NEXT:         dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_H)), N,
  // CHECK-NEXT:         beta_H,
  // CHECK-NEXT:         dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_H)), N))) {
  // CHECK-NEXT: }
  switch (int stat = cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N)){
  }


  return 0;
}

// CHECK: int foo() try {
// CHECK-NEXT:   return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(
// CHECK-NEXT:       handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
// CHECK-NEXT:       dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_H)), N,
// CHECK-NEXT:       dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_H)), N, beta_H,
// CHECK-NEXT:       dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_H)), N));
// CHECK-NEXT: }
int foo() {
  return cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);
}

// CHECK: void foo2() {
// CHECK-NEXT:   oneapi::mkl::blas::column_major::symv(
// CHECK-NEXT:       handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
// CHECK-NEXT:       dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_H)), N,
// CHECK-NEXT:       dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_H)), N, beta_H,
// CHECK-NEXT:       dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_H)), N);
// CHECK-NEXT: }
void foo2() {
  cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);
}
