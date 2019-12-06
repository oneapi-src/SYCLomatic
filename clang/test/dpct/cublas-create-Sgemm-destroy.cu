// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-create-Sgemm-destroy.dp.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <cstdio>
// CHECK: #include <mkl_blas_sycl.hpp>
// CHECK-NEXT: #include <mkl_lapack_sycl.hpp>
// CHECK-NEXT: #include <mkl_sycl_types.hpp>
#include <cstdio>
#include "cublas_v2.h"
#include <cuda_runtime.h>

void foo (cublasStatus_t s){
}
cublasStatus_t bar (cublasStatus_t s){
  return s;
}

// CHECK: extern cl::sycl::queue handle2;
extern cublasHandle_t handle2;

int main() {
  // CHECK: int status;
  // CHECK-NEXT: cl::sycl::queue handle;
  // CHECK-NEXT: status = 0;
  // CHECK-NEXT: if (status != 0) {
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }
  int N = 275;
  float *d_A_S = 0;
  float *d_B_S = 0;
  float *d_C_S = 0;
  float alpha_S = 1.0f;
  float beta_S = 0.0f;
  int trans0 = 0;
  int trans1 = 1;
  int trans2 = 2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK: status = (mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_S), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, *(&beta_S), buffer_ct{{[0-9]+}}, N), 0);
  // CHECK: mkl::blas::gemm(handle, (((int)transpose_ct1)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct1)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), N, N, N, *(&alpha_S), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, *(&beta_S), buffer_ct{{[0-9]+}}, N);
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  double *d_A_D = 0;
  double *d_B_D = 0;
  double *d_C_D = 0;
  double alpha_D = 1.0;
  double beta_D = 0.0;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK: status = (mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_D), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, *(&beta_D), buffer_ct{{[0-9]+}}, N), 0);
  // CHECK: mkl::blas::gemm(handle, (((int)transpose_ct1)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct1)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), N, N, N, *(&alpha_D), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, *(&beta_D), buffer_ct{{[0-9]+}}, N);
  status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  cublasDgemm(handle, (cublasOperation_t)trans2, (cublasOperation_t)2, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);



  // CHECK: for (;;) {
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::blas::gemm(handle, mkl::transpose::trans, mkl::transpose::trans, N, N, N, *(&alpha_S), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, *(&beta_S), buffer_ct{{[0-9]+}}, N), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: beta_S = beta_S + 1;
  // CHECK-NEXT: }
  // CHECK-NEXT: alpha_S = alpha_S + 1;
  for (;;) {
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
    beta_S = beta_S + 1;
  }
  alpha_S = alpha_S + 1;

  // CHECK: for (;;) {
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::trans, mkl::transpose::trans, N, N, N, *(&alpha_S), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, *(&beta_S), buffer_ct{{[0-9]+}}, N);
  // CHECK-NEXT: }
  // CHECK-NEXT: beta_S = beta_S + 1;
  // CHECK-NEXT: }
  // CHECK-NEXT: alpha_S = alpha_S + 1;
  for (;;) {
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
    beta_S = beta_S + 1;
  }
  alpha_S = alpha_S + 1;


  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: foo(bar((mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_S), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, *(&beta_S), buffer_ct{{[0-9]+}}, N), 0)));
  foo(bar(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)));

  // CHECK: status = 0;
  // CHECK-NEXT: return 0;
  status = cublasDestroy(handle);
  cublasDestroy(handle);
  return 0;
}
