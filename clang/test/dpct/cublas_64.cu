// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8
// RUN: dpct --format-range=none --usm-level=none --out-root %T/cublas_64 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/cublas_64/cublas_64.dp.cpp --match-full-lines %s

#include "cublas_v2.h"

void foo() {
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasOperation_t transa;
  cublasOperation_t transb;
  int64_t m;
  int64_t n;
  int64_t k;
  const float *alpha_s;
  const double *alpha_d;
  const float2 *alpha_c;
  const double2 *alpha_z;
  const float *A_s;
  const double *A_d;
  const float2 *A_c;
  const double2 *A_z;
  int64_t lda;
  const float *B_s;
  const double *B_d;
  const float2 *B_c;
  const double2 *B_z;
  int64_t ldb;
  const float *beta_s;
  const double *beta_d;
  const float2 *beta_c;
  const double2 *beta_z;
  float *C_s;
  double *C_d;
  float2 *C_c;
  double2 *C_z;
  int64_t ldc;
  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(*handle, transa, transb, m, n, k, dpct::get_value(alpha_s, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(B_s)), ldb, dpct::get_value(beta_s, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(*handle, transa, transb, m, n, k, dpct::get_value(alpha_d, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(B_d)), ldb, dpct::get_value(beta_d, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(*handle, transa, transb, m, n, k, dpct::get_value(alpha_c, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(*handle, transa, transb, m, n, k, dpct::get_value(alpha_z, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  status = cublasSgemm_64(handle, transa, transb, m, n, k, alpha_s, A_s, lda, B_s, ldb, beta_s, C_s, ldc);
  status = cublasDgemm_64(handle, transa, transb, m, n, k, alpha_d, A_d, lda, B_d, ldb, beta_d, C_d, ldc);
  status = cublasCgemm_64(handle, transa, transb, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZgemm_64(handle, transa, transb, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  cublasFillMode_t uplo;
  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(*handle, uplo, transa, n, k, dpct::get_value(alpha_s, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::get_value(B_s, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(*handle, uplo, transa, n, k, dpct::get_value(alpha_d, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::get_value(B_d, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(*handle, uplo, transa, n, k, dpct::get_value(alpha_c, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::get_value(B_c, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(*handle, uplo, transa, n, k, dpct::get_value(alpha_z, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::get_value(B_z, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  status = cublasSsyrk_64(handle, uplo, transa, n, k, alpha_s, A_s, lda, B_s, C_s, ldc);
  status = cublasDsyrk_64(handle, uplo, transa, n, k, alpha_d, A_d, lda, B_d, C_d, ldc);
  status = cublasCsyrk_64(handle, uplo, transa, n, k, alpha_c, A_c, lda, B_c, C_c, ldc);
  status = cublasZsyrk_64(handle, uplo, transa, n, k, alpha_z, A_z, lda, B_z, C_z, ldc);
}
