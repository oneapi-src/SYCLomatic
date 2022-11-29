// RUN: cat %s > %T/cublas-lambda.cu
// RUN: cd %T
// RUN: dpct --no-cl-namespace-inline --usm-level=none -out-root %T/cublas-lambda cublas-lambda.cu --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-lambda/cublas-lambda.dp.cpp --match-full-lines cublas-lambda.cu
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
float *d_A_S = 0;
float *d_B_S = 0;
float *d_C_S = 0;
float alpha_S = 1.0f;
float beta_S = 0.0f;

int main() {
  cublasStatus_t status;
  cublasHandle_t handle;
  // CHECK: handle = &dpct::get_default_queue();
  cublasCreate(&handle);

  // CHECK: {
  // CHECK-NEXT:   auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
  // CHECK-NEXT:   auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
  // CHECK-NEXT:   auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
  // CHECK-NEXT:   oneapi::mkl::blas::column_major::gemm(
  // CHECK-NEXT:       *handle, oneapi::mkl::transpose::nontrans,
  // CHECK-NEXT:       oneapi::mkl::transpose::nontrans, N, N, N, alpha_S, d_A_S_buf_ct{{[0-9]+}}, N,
  // CHECK-NEXT:       d_B_S_buf_ct{{[0-9]+}}, N, beta_S, d_C_S_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT: }
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error
  // CHECK-NEXT: codes. 0 is used instead of an error code in an if statement. You may need to
  // CHECK-NEXT: rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (0) {
  // CHECK-NEXT: }
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the
  // CHECK-NEXT: lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: else if ([&]() {
  // CHECK-NEXT:            auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
  // CHECK-NEXT:            auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
  // CHECK-NEXT:            auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
  // CHECK-NEXT:            oneapi::mkl::blas::column_major::gemm(
  // CHECK-NEXT:                *handle, oneapi::mkl::transpose::nontrans,
  // CHECK-NEXT:                oneapi::mkl::transpose::nontrans, N, N, N, alpha_S,
  // CHECK-NEXT:                d_A_S_buf_ct{{[0-9]+}}, N, d_B_S_buf_ct{{[0-9]+}}, N, beta_S, d_C_S_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT:            return 0;
  // CHECK-NEXT:          }()) {
  // CHECK-NEXT: }
  if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)) {
  }
  else if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)) {
  }


  // CHECK: {
  // CHECK-NEXT:   auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
  // CHECK-NEXT:   auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
  // CHECK-NEXT:   auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
  // CHECK-NEXT:   oneapi::mkl::blas::column_major::gemm(
  // CHECK-NEXT:       *handle, oneapi::mkl::transpose::nontrans,
  // CHECK-NEXT:       oneapi::mkl::transpose::nontrans, N, N, N, alpha_S, d_A_S_buf_ct{{[0-9]+}}, N,
  // CHECK-NEXT:       d_B_S_buf_ct{{[0-9]+}}, N, beta_S, d_C_S_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT: }
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error
  // CHECK-NEXT: codes. 0 is used instead of an error code in an if statement. You may need to
  // CHECK-NEXT: rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (int stat = 0) {
  // CHECK-NEXT: }
  if(int stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){
  }


  // CHECK: {
  // CHECK-NEXT:   auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
  // CHECK-NEXT:   auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
  // CHECK-NEXT:   auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
  // CHECK-NEXT:   oneapi::mkl::blas::column_major::gemm(
  // CHECK-NEXT:       *handle, oneapi::mkl::transpose::nontrans,
  // CHECK-NEXT:       oneapi::mkl::transpose::nontrans, N, N, N, alpha_S, d_A_S_buf_ct{{[0-9]+}}, N,
  // CHECK-NEXT:       d_B_S_buf_ct{{[0-9]+}}, N, beta_S, d_C_S_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT: }
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error
  // CHECK-NEXT: codes. 0 is used instead of an error code in a for statement. You may need to
  // CHECK-NEXT: rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: for (0;;) {
  // CHECK-NEXT: }
  for(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);;){
  }

  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the
  // CHECK-NEXT: lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: for (; [&]() {
  // CHECK-NEXT: auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(
  // CHECK-NEXT:     *handle, oneapi::mkl::transpose::nontrans,
  // CHECK-NEXT:     oneapi::mkl::transpose::nontrans, N, N, N, alpha_S, d_A_S_buf_ct{{[0-9]+}},
  // CHECK-NEXT:     N, d_B_S_buf_ct{{[0-9]+}}, N, beta_S, d_C_S_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();) {
  // CHECK-NEXT: }
  for(;cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);){
  }

  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the
  // CHECK-NEXT: lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: while ([&]() {
  // CHECK-NEXT: auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(
  // CHECK-NEXT:     *handle, oneapi::mkl::transpose::nontrans,
  // CHECK-NEXT:     oneapi::mkl::transpose::nontrans, N, N, N, alpha_S, d_A_S_buf_ct{{[0-9]+}}, N,
  // CHECK-NEXT:     d_B_S_buf_ct{{[0-9]+}}, N, beta_S, d_C_S_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }() != 0) {
  // CHECK-NEXT: }
  while(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)!=0){
  }



  // CHECK: do{
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the
  // CHECK-NEXT: lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: } while ([&]() {
  // CHECK-NEXT: auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(
  // CHECK-NEXT:     *handle, oneapi::mkl::transpose::nontrans,
  // CHECK-NEXT:     oneapi::mkl::transpose::nontrans, N, N, N, alpha_S, d_A_S_buf_ct{{[0-9]+}}, N,
  // CHECK-NEXT:     d_B_S_buf_ct{{[0-9]+}}, N, beta_S, d_C_S_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }());
  do{
  }while(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N));


  // CHECK: {
  // CHECK-NEXT:   auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
  // CHECK-NEXT:   auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
  // CHECK-NEXT:   auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
  // CHECK-NEXT:   oneapi::mkl::blas::column_major::gemm(
  // CHECK-NEXT:       *handle, oneapi::mkl::transpose::nontrans,
  // CHECK-NEXT:       oneapi::mkl::transpose::nontrans, N, N, N, alpha_S, d_A_S_buf_ct{{[0-9]+}}, N,
  // CHECK-NEXT:       d_B_S_buf_ct{{[0-9]+}}, N, beta_S, d_C_S_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT: }
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error
  // CHECK-NEXT: codes. 0 is used instead of an error code in a switch statement. You may need
  // CHECK-NEXT: to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: switch (int stat = 0) {
  // CHECK-NEXT: }
  switch (int stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){
  }


  return 0;
}

// CHECK:int foo() try {
// CHECK-NEXT:  auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
// CHECK-NEXT:  auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
// CHECK-NEXT:  auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
// CHECK-NEXT:  oneapi::mkl::blas::column_major::gemm(
// CHECK-NEXT:      *handle, oneapi::mkl::transpose::nontrans,
// CHECK-NEXT:      oneapi::mkl::transpose::nontrans, N, N, N, alpha_S, d_A_S_buf_ct{{[0-9]+}}, N,
// CHECK-NEXT:      d_B_S_buf_ct{{[0-9]+}}, N, beta_S, d_C_S_buf_ct{{[0-9]+}}, N);
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error
// CHECK-NEXT:  codes. 0 is used instead of an error code in a return statement. You may need
// CHECK-NEXT:  to rewrite this code.
// CHECK-NEXT:  */
// CHECK-NEXT:  return 0;
// CHECK-NEXT:}
int foo() {
  return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
}

// CHECK:void foo2() {
// CHECK-NEXT:  auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
// CHECK-NEXT:  auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
// CHECK-NEXT:  auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
// CHECK-NEXT:  oneapi::mkl::blas::column_major::gemm(
// CHECK-NEXT:      *handle, oneapi::mkl::transpose::nontrans,
// CHECK-NEXT:      oneapi::mkl::transpose::nontrans, N, N, N, alpha_S, d_A_S_buf_ct{{[0-9]+}}, N,
// CHECK-NEXT:      d_B_S_buf_ct{{[0-9]+}}, N, beta_S, d_C_S_buf_ct{{[0-9]+}}, N);
// CHECK-NEXT:}
void foo2() {
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
}