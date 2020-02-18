// RUN: cat %s > %T/cublas-lambda.cu
// RUN: cd %T
// RUN: dpct --no-cl-namespace-inline -out-root %T cublas-lambda.cu --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-lambda.dp.cpp --match-full-lines cublas-lambda.cu
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <cstdio>
// CHECK: #include <mkl_blas_sycl.hpp>
// CHECK-NEXT: #include <mkl_lapack_sycl.hpp>
// CHECK-NEXT: #include <mkl_sycl_types.hpp>
#include <cstdio>
#include "cublas_v2.h"
#include <cuda_runtime.h>


int main() {
  cublasStatus_t status;
  cublasHandle_t handle;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasCreate was removed, because this call is
  // CHECK-NEXT: redundant in DPC++.
  // CHECK-NEXT: */
  cublasCreate(&handle);

  int N = 275;
  float *d_A_S = 0;
  float *d_B_S = 0;
  float *d_C_S = 0;
  float alpha_S = 1.0f;
  float beta_S = 0.0f;

  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated api does not return error code. 0 is returned in the
  // CHECK-NEXT: lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if ([&]() {
  // CHECK-NEXT: auto d_A_S_buff_ct1 =
  // CHECK-NEXT:     dpct::mem_mgr::instance().get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buff_ct1 =
  // CHECK-NEXT:     dpct::mem_mgr::instance().get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buff_ct1 =
  // CHECK-NEXT:     dpct::mem_mgr::instance().get_buffer<float>(d_C_S);
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans,
  // CHECK-NEXT:                 mkl::transpose::nontrans, N, N, N, *(&alpha_S),
  // CHECK-NEXT:                 d_A_S_buff_ct1, N, d_B_S_buff_ct1, N, *(&beta_S),
  // CHECK-NEXT:                 d_C_S_buff_ct1, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }()) {
  // CHECK-NEXT: }
  if(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){
  }


  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated api does not return error code. 0 is returned in the
  // CHECK-NEXT: lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (int stat = [&]() {
  // CHECK-NEXT: auto d_A_S_buff_ct1 =
  // CHECK-NEXT:     dpct::mem_mgr::instance().get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buff_ct1 =
  // CHECK-NEXT:     dpct::mem_mgr::instance().get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buff_ct1 =
  // CHECK-NEXT:     dpct::mem_mgr::instance().get_buffer<float>(d_C_S);
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans,
  // CHECK-NEXT:                 mkl::transpose::nontrans, N, N, N, *(&alpha_S),
  // CHECK-NEXT:                 d_A_S_buff_ct1, N, d_B_S_buff_ct1, N, *(&beta_S),
  // CHECK-NEXT:                 d_C_S_buff_ct1, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }()) {
  // CHECK-NEXT: }
  if(int stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){
  }


  // CHECK: for ([&]() {
  // CHECK-NEXT: auto d_A_S_buff_ct1 =
  // CHECK-NEXT:     dpct::mem_mgr::instance().get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buff_ct1 =
  // CHECK-NEXT:     dpct::mem_mgr::instance().get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buff_ct1 =
  // CHECK-NEXT:     dpct::mem_mgr::instance().get_buffer<float>(d_C_S);
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans,
  // CHECK-NEXT:                 mkl::transpose::nontrans, N, N, N, *(&alpha_S),
  // CHECK-NEXT:                 d_A_S_buff_ct1, N, d_B_S_buff_ct1, N, *(&beta_S),
  // CHECK-NEXT:                 d_C_S_buff_ct1, N);
  // CHECK-NEXT: }();
  // CHECK-NEXT: ;) {
  // CHECK-NEXT: }
  for(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);;){
  }


  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated api does not return error code. 0 is returned in the
  // CHECK-NEXT: lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: while ([&]() {
  // CHECK-NEXT: auto d_A_S_buff_ct1 = dpct::mem_mgr::instance().get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buff_ct1 = dpct::mem_mgr::instance().get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buff_ct1 = dpct::mem_mgr::instance().get_buffer<float>(d_C_S);
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans,
  // CHECK-NEXT:                 N, N, N, *(&alpha_S), d_A_S_buff_ct1, N, d_B_S_buff_ct1, N,
  // CHECK-NEXT:                 *(&beta_S), d_C_S_buff_ct1, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }() != 0) {
  // CHECK-NEXT: }
  while(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)!=0){
  }



  // CHECK: do{
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated api does not return error code. 0 is returned in the
  // CHECK-NEXT: lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: } while ([&]() {
  // CHECK-NEXT: auto d_A_S_buff_ct1 = dpct::mem_mgr::instance().get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buff_ct1 = dpct::mem_mgr::instance().get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buff_ct1 = dpct::mem_mgr::instance().get_buffer<float>(d_C_S);
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans,
  // CHECK-NEXT:                 N, N, N, *(&alpha_S), d_A_S_buff_ct1, N, d_B_S_buff_ct1, N,
  // CHECK-NEXT:                 *(&beta_S), d_C_S_buff_ct1, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }());
  do{
  }while(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N));


  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated api does not return error code. 0 is returned in the
  // CHECK-NEXT: lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: switch (int stat = [&]() {
  // CHECK-NEXT: auto d_A_S_buff_ct1 = dpct::mem_mgr::instance().get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buff_ct1 = dpct::mem_mgr::instance().get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buff_ct1 = dpct::mem_mgr::instance().get_buffer<float>(d_C_S);
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans,
  // CHECK-NEXT:                 N, N, N, *(&alpha_S), d_A_S_buff_ct1, N, d_B_S_buff_ct1, N,
  // CHECK-NEXT:                 *(&beta_S), d_C_S_buff_ct1, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }()) {}
  switch (int stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){
  }


  return 0;
}
