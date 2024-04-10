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
  // CHECK: {
  // CHECK-NEXT:   auto d_A_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_H);
  // CHECK-NEXT:   auto d_B_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_H);
  // CHECK-NEXT:   auto d_C_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_H);
  // CHECK-NEXT:   oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:       handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:       d_A_H_buf_ct{{[0-9]+}}, N, d_B_H_buf_ct{{[0-9]+}}, N, beta_H, d_C_H_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT: }
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error
  // CHECK-NEXT: codes. 0 is used instead of an error code in an if statement. You may need to
  // CHECK-NEXT: rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (0) {
  // CHECK-NEXT: }
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the
  // CHECK-NEXT: lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: else if ([&]() {
  // CHECK-NEXT:            auto d_A_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_H);
  // CHECK-NEXT:            auto d_B_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_H);
  // CHECK-NEXT:            auto d_C_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_H);
  // CHECK-NEXT:            oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:                handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:                d_A_H_buf_ct{{[0-9]+}}, N, d_B_H_buf_ct{{[0-9]+}}, N, beta_H, d_C_H_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT:            return 0;
  // CHECK-NEXT:          }()) {
  // CHECK-NEXT: }
  if (cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N)) {
  }
  else if (cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N)) {
  }


  // CHECK: {
  // CHECK-NEXT:   auto d_A_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_H);
  // CHECK-NEXT:   auto d_B_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_H);
  // CHECK-NEXT:   auto d_C_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_H);
  // CHECK-NEXT:   oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:       handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:       d_A_H_buf_ct{{[0-9]+}}, N, d_B_H_buf_ct{{[0-9]+}}, N, beta_H, d_C_H_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT: }
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error
  // CHECK-NEXT: codes. 0 is used instead of an error code in an if statement. You may need to
  // CHECK-NEXT: rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (int stat = 0) {
  // CHECK-NEXT: }
  if(int stat = cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N)){
  }


  // CHECK: {
  // CHECK-NEXT:   auto d_A_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_H);
  // CHECK-NEXT:   auto d_B_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_H);
  // CHECK-NEXT:   auto d_C_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_H);
  // CHECK-NEXT:   oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:       handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:       d_A_H_buf_ct{{[0-9]+}}, N, d_B_H_buf_ct{{[0-9]+}}, N, beta_H, d_C_H_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT: }
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error
  // CHECK-NEXT: codes. 0 is used instead of an error code in a for statement. You may need to
  // CHECK-NEXT: rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: for (0;;) {
  // CHECK-NEXT: }
  for(cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);;){
  }

  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the
  // CHECK-NEXT: lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: for (; [&]() {
  // CHECK-NEXT:        auto d_A_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_H);
  // CHECK-NEXT:        auto d_B_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_H);
  // CHECK-NEXT:        auto d_C_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_H);
  // CHECK-NEXT:        oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:            handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:            d_A_H_buf_ct{{[0-9]+}}, N, d_B_H_buf_ct{{[0-9]+}}, N, beta_H, d_C_H_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT:        return 0;
  // CHECK-NEXT:      }();) {
  // CHECK-NEXT: }
  for(;cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);){
  }

  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the
  // CHECK-NEXT: lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: while ([&]() {
  // CHECK-NEXT:   auto d_A_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_H);
  // CHECK-NEXT:   auto d_B_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_H);
  // CHECK-NEXT:   auto d_C_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_H);
  // CHECK-NEXT:   oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:       handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:       d_A_H_buf_ct{{[0-9]+}}, N, d_B_H_buf_ct{{[0-9]+}}, N, beta_H, d_C_H_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT:   return 0;
  // CHECK-NEXT: }() != 0) {
  // CHECK-NEXT: }
  while(cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N)!=0){
  }



  // CHECK: do{
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the
  // CHECK-NEXT: lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: } while ([&]() {
  // CHECK-NEXT:   auto d_A_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_H);
  // CHECK-NEXT:   auto d_B_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_H);
  // CHECK-NEXT:   auto d_C_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_H);
  // CHECK-NEXT:   oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:       handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:       d_A_H_buf_ct{{[0-9]+}}, N, d_B_H_buf_ct{{[0-9]+}}, N, beta_H, d_C_H_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT:   return 0;
  // CHECK-NEXT: }());
  do{
  }while(cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N));


  // CHECK: {
  // CHECK-NEXT:   auto d_A_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_H);
  // CHECK-NEXT:   auto d_B_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_H);
  // CHECK-NEXT:   auto d_C_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_H);
  // CHECK-NEXT:   oneapi::mkl::blas::column_major::symv(
  // CHECK-NEXT:       handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H,
  // CHECK-NEXT:       d_A_H_buf_ct{{[0-9]+}}, N, d_B_H_buf_ct{{[0-9]+}}, N, beta_H, d_C_H_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT: }
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error
  // CHECK-NEXT: codes. 0 is used instead of an error code in a switch statement. You may need
  // CHECK-NEXT: to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: switch (int stat = 0) {
  // CHECK-NEXT: }
  switch (int stat = cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N)){
  }


  return 0;
}

// CHECK:int foo() try {
// CHECK-NEXT:  auto d_A_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_H);
// CHECK-NEXT:  auto d_B_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_H);
// CHECK-NEXT:  auto d_C_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_H);
// CHECK-NEXT:  oneapi::mkl::blas::column_major::symv(
// CHECK-NEXT:      handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H, d_A_H_buf_ct{{[0-9]+}},
// CHECK-NEXT:      N, d_B_H_buf_ct{{[0-9]+}}, N, beta_H, d_C_H_buf_ct{{[0-9]+}}, N);
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error
// CHECK-NEXT:  codes. 0 is used instead of an error code in a return statement. You may need
// CHECK-NEXT:  to rewrite this code.
// CHECK-NEXT:  */
// CHECK-NEXT:  return 0;
// CHECK-NEXT:}
int foo() {
  return cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);
}

// CHECK:void foo2() {
// CHECK-NEXT:  auto d_A_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_H);
// CHECK-NEXT:  auto d_B_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_H);
// CHECK-NEXT:  auto d_C_H_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_H);
// CHECK-NEXT:  oneapi::mkl::blas::column_major::symv(
// CHECK-NEXT:      handle->get_queue(), oneapi::mkl::uplo::upper, N, alpha_H, d_A_H_buf_ct{{[0-9]+}},
// CHECK-NEXT:      N, d_B_H_buf_ct{{[0-9]+}}, N, beta_H, d_C_H_buf_ct{{[0-9]+}}, N);
// CHECK-NEXT:}
void foo2() {
  cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);
}
