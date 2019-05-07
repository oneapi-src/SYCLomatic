// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/cublas-create-Sgemm-destroy.sycl.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <syclct/syclct.hpp>
// CHECK-NEXT: #include <cstdio>
// CHECK: #include <DPCPP_blas_TEMP.h>
// CHECK: #include <complex>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
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
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::Sgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N), 0);
  // CHECK-NEXT: mkl::Sgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  double *d_A_D = 0;
  double *d_B_D = 0;
  double *d_C_D = 0;
  double alpha_D = 1.0;
  double beta_D = 0.0;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::Dgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N), 0);
  // CHECK-NEXT: mkl::Dgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  float2 *d_A_C = 0;
  float2 *d_B_C = 0;
  float2 *d_C_C = 0;
  float2 alpha_C = make_float2(1, 0);
  float2 beta_C = make_float2(0, 0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::Cgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N), 0);
  // CHECK-NEXT: mkl::Cgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N);
  status = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N);
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N);
  double2 *d_A_Z = 0;
  double2 *d_B_Z = 0;
  double2 *d_C_Z = 0;
  double2 alpha_Z = make_double2(1, 0);
  double2 beta_Z = make_double2(0, 0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::Zgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, &beta_Z, d_C_Z, N), 0);
  // CHECK-NEXT: mkl::Zgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, &beta_Z, d_C_Z, N);
  status = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, &beta_Z, d_C_Z, N);
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, &beta_Z, d_C_Z, N);
  cuComplex *d_A_C_2 = 0;
  cuComplex *d_B_C_2 = 0;
  cuComplex *d_C_C_2 = 0;
  cuComplex alpha_C_2 = make_float2(1, 0);
  cuComplex beta_C_2 = make_float2(0, 0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::Cgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, &alpha_C_2, d_A_C_2, N, d_B_C_2, N, &beta_C_2, d_C_C_2, N), 0);
  // CHECK-NEXT: mkl::Cgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, &alpha_C_2, d_A_C_2, N, d_B_C_2, N, &beta_C_2, d_C_C_2, N);
  status = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C_2, d_A_C_2, N, d_B_C_2, N, &beta_C_2, d_C_C_2, N);
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C_2, d_A_C_2, N, d_B_C_2, N, &beta_C_2, d_C_C_2, N);
  cuDoubleComplex *d_A_Z_2 = 0;
  cuDoubleComplex *d_B_Z_2 = 0;
  cuDoubleComplex *d_C_Z_2 = 0;
  cuDoubleComplex alpha_Z_2 = make_double2(1, 0);
  cuDoubleComplex beta_Z_2 = make_double2(0, 0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::Zgemm(handle, mkl::transpose::trans, mkl::transpose::trans, N, N, N, &alpha_Z_2, d_A_Z_2, N, d_B_Z_2, N, &beta_Z_2, d_C_Z_2, N), 0);
  // CHECK-NEXT: mkl::Zgemm(handle, mkl::transpose::conjtrans, mkl::transpose::conjtrans, N, N, N, &alpha_Z_2, d_A_Z_2, N, d_B_Z_2, N, &beta_Z_2, d_C_Z_2, N);
  status = cublasZgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha_Z_2, d_A_Z_2, N, d_B_Z_2, N, &beta_Z_2, d_C_Z_2, N);
  cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_Z_2, d_A_Z_2, N, d_B_Z_2, N, &beta_Z_2, d_C_Z_2, N);
  // CHECK: status = 0;
  // CHECK-NEXT: return 0;
  status = cublasDestroy(handle);
  cublasDestroy(handle);
  return 0;
}