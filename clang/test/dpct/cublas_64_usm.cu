// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8
// RUN: dpct --format-range=none --out-root %T/cublas_64_usm %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/cublas_64_usm/cublas_64_usm.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cublas_64_usm/cublas_64_usm.dp.cpp -o %T/cublas_64_usm/cublas_64_usm.dp.o %}

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
  cublasFillMode_t uplo;
  cublasSideMode_t side;
  cublasDiagType_t diag;
  std::int64_t result;

  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::out_mem_int64_t res(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, A_s, lda, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::out_mem_int64_t res(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, A_d, lda, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::out_mem_int64_t res(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, (std::complex<float>*)A_c, lda, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::out_mem_int64_t res(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, (std::complex<double>*)A_z, lda, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasIsamax_64(handle, n, A_s, lda, &result);
  status = cublasIdamax_64(handle, n, A_d, lda, &result);
  status = cublasIcamax_64(handle, n, A_c, lda, &result);
  status = cublasIzamax_64(handle, n, A_z, lda, &result);

  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::out_mem_int64_t res(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, A_s, lda, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::out_mem_int64_t res(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, A_d, lda, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::out_mem_int64_t res(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, (std::complex<float>*)A_c, lda, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::out_mem_int64_t res(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, (std::complex<double>*)A_z, lda, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasIsamin_64(handle, n, A_s, lda, &result);
  status = cublasIdamin_64(handle, n, A_d, lda, &result);
  status = cublasIcamin_64(handle, n, A_c, lda, &result);
  status = cublasIzamin_64(handle, n, A_z, lda, &result);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha_s, handle->get_queue()), A_s, lda, B_s, ldb, dpct::get_value(beta_s, handle->get_queue()), C_s, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha_d, handle->get_queue()), A_d, lda, B_d, ldb, dpct::get_value(beta_d, handle->get_queue()), C_d, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha_c, handle->get_queue()), (std::complex<float>*)A_c, lda, (std::complex<float>*)B_c, ldb, dpct::get_value(beta_c, handle->get_queue()), (std::complex<float>*)C_c, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha_z, handle->get_queue()), (std::complex<double>*)A_z, lda, (std::complex<double>*)B_z, ldb, dpct::get_value(beta_z, handle->get_queue()), (std::complex<double>*)C_z, ldc));
  status = cublasSgemm_64(handle, transa, transb, m, n, k, alpha_s, A_s, lda, B_s, ldb, beta_s, C_s, ldc);
  status = cublasDgemm_64(handle, transa, transb, m, n, k, alpha_d, A_d, lda, B_d, ldb, beta_d, C_d, ldc);
  status = cublasCgemm_64(handle, transa, transb, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZgemm_64(handle, transa, transb, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_s, handle->get_queue()), A_s, lda, dpct::get_value(beta_s, handle->get_queue()), C_s, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_d, handle->get_queue()), A_d, lda, dpct::get_value(beta_d, handle->get_queue()), C_d, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_c, handle->get_queue()), (std::complex<float>*)A_c, lda, dpct::get_value(beta_c, handle->get_queue()), (std::complex<float>*)C_c, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_z, handle->get_queue()), (std::complex<double>*)A_z, lda, dpct::get_value(beta_z, handle->get_queue()), (std::complex<double>*)C_z, ldc));
  status = cublasSsyrk_64(handle, uplo, transa, n, k, alpha_s, A_s, lda, beta_s, C_s, ldc);
  status = cublasDsyrk_64(handle, uplo, transa, n, k, alpha_d, A_d, lda, beta_d, C_d, ldc);
  status = cublasCsyrk_64(handle, uplo, transa, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);
  status = cublasZsyrk_64(handle, uplo, transa, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(handle->get_queue(), side, uplo, m, n, dpct::get_value(alpha_s, handle->get_queue()), A_s, lda, B_s, ldb, dpct::get_value(beta_s, handle->get_queue()), C_s, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(handle->get_queue(), side, uplo, m, n, dpct::get_value(alpha_d, handle->get_queue()), A_d, lda, B_d, ldb, dpct::get_value(beta_d, handle->get_queue()), C_d, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(handle->get_queue(), side, uplo, m, n, dpct::get_value(alpha_c, handle->get_queue()), (std::complex<float>*)A_c, lda, (std::complex<float>*)B_c, ldb, dpct::get_value(beta_c, handle->get_queue()), (std::complex<float>*)C_c, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(handle->get_queue(), side, uplo, m, n, dpct::get_value(alpha_z, handle->get_queue()), (std::complex<double>*)A_z, lda, (std::complex<double>*)B_z, ldb, dpct::get_value(beta_z, handle->get_queue()), (std::complex<double>*)C_z, ldc));
  status = cublasSsymm_64(handle, side, uplo, m, n, alpha_s, A_s, lda, B_s, ldb, beta_s, C_s, ldc);
  status = cublasDsymm_64(handle, side, uplo, m, n, alpha_d, A_d, lda, B_d, ldb, beta_d, C_d, ldc);
  status = cublasCsymm_64(handle, side, uplo, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZsymm_64(handle, side, uplo, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(handle->get_queue(), side, uplo, transa, diag, m, n, dpct::get_value(alpha_s, handle->get_queue()), A_s, lda, C_s, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(handle->get_queue(), side, uplo, transa, diag, m, n, dpct::get_value(alpha_d, handle->get_queue()), A_d, lda, C_d, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(handle->get_queue(), side, uplo, transa, diag, m, n, dpct::get_value(alpha_c, handle->get_queue()), (std::complex<float>*)A_c, lda, (std::complex<float>*)C_c, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(handle->get_queue(), side, uplo, transa, diag, m, n, dpct::get_value(alpha_z, handle->get_queue()), (std::complex<double>*)A_z, lda, (std::complex<double>*)C_z, ldc));
  status = cublasStrsm_64(handle, side, uplo, transa, diag, m, n, alpha_s, A_s, lda, C_s, ldc);
  status = cublasDtrsm_64(handle, side, uplo, transa, diag, m, n, alpha_d, A_d, lda, C_d, ldc);
  status = cublasCtrsm_64(handle, side, uplo, transa, diag, m, n, alpha_c, A_c, lda, C_c, ldc);
  status = cublasZtrsm_64(handle, side, uplo, transa, diag, m, n, alpha_z, A_z, lda, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemm(handle->get_queue(), side, uplo, m, n, dpct::get_value(alpha_c, handle->get_queue()), (std::complex<float>*)A_c, lda, (std::complex<float>*)B_c, ldb, dpct::get_value(beta_c, handle->get_queue()), (std::complex<float>*)C_c, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemm(handle->get_queue(), side, uplo, m, n, dpct::get_value(alpha_z, handle->get_queue()), (std::complex<double>*)A_z, lda, (std::complex<double>*)B_z, ldb, dpct::get_value(beta_z, handle->get_queue()), (std::complex<double>*)C_z, ldc));
  status = cublasChemm_64(handle, side, uplo, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZhemm_64(handle, side, uplo, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::herk(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_s, handle->get_queue()), (std::complex<float>*)A_c, lda, dpct::get_value(beta_s, handle->get_queue()), (std::complex<float>*)C_c, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::herk(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_d, handle->get_queue()), (std::complex<double>*)A_z, lda, dpct::get_value(beta_d, handle->get_queue()), (std::complex<double>*)C_z, ldc));
  status = cublasCherk_64(handle, uplo, transa, n, k, alpha_s, A_c, lda, beta_s, C_c, ldc);
  status = cublasZherk_64(handle, uplo, transa, n, k, alpha_d, A_z, lda, beta_d, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_s, handle->get_queue()), A_s, lda, B_s, ldb, dpct::get_value(beta_s, handle->get_queue()), C_s, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_d, handle->get_queue()), A_d, lda, B_d, ldb, dpct::get_value(beta_d, handle->get_queue()), C_d, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_c, handle->get_queue()), (std::complex<float>*)A_c, lda, (std::complex<float>*)B_c, ldb, dpct::get_value(beta_c, handle->get_queue()), (std::complex<float>*)C_c, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_z, handle->get_queue()), (std::complex<double>*)A_z, lda, (std::complex<double>*)B_z, ldb, dpct::get_value(beta_z, handle->get_queue()), (std::complex<double>*)C_z, ldc));
  status = cublasSsyr2k_64(handle, uplo, transa, n, k, alpha_s, A_s, lda, B_s, ldb, beta_s, C_s, ldc);
  status = cublasDsyr2k_64(handle, uplo, transa, n, k, alpha_d, A_d, lda, B_d, ldb, beta_d, C_d, ldc);
  status = cublasCsyr2k_64(handle, uplo, transa, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZsyr2k_64(handle, uplo, transa, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2k(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_c, handle->get_queue()), (std::complex<float>*)A_c, lda, (std::complex<float>*)B_c, ldb, dpct::get_value(beta_s, handle->get_queue()), (std::complex<float>*)C_c, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2k(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_z, handle->get_queue()), (std::complex<double>*)A_z, lda, (std::complex<double>*)B_z, ldb, dpct::get_value(beta_d, handle->get_queue()), (std::complex<double>*)C_z, ldc));
  status = cublasCher2k_64(handle, uplo, transa, n, k, alpha_c, A_c, lda, B_c, ldb, beta_s, C_c, ldc);
  status = cublasZher2k_64(handle, uplo, transa, n, k, alpha_z, A_z, lda, B_z, ldb, beta_d, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha_s, handle->get_queue()), A_s, lda, dpct::get_value(beta_s, handle->get_queue()), B_s, ldb, C_s, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha_d, handle->get_queue()), A_d, lda, dpct::get_value(beta_d, handle->get_queue()), B_d, ldb, C_d, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha_c, handle->get_queue()), (std::complex<float>*)A_c, lda, dpct::get_value(beta_c, handle->get_queue()), (std::complex<float>*)B_c, ldb, (std::complex<float>*)C_c, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha_z, handle->get_queue()), (std::complex<double>*)A_z, lda, dpct::get_value(beta_z, handle->get_queue()), (std::complex<double>*)B_z, ldb, (std::complex<double>*)C_z, ldc));
  status = cublasSgeam_64(handle, transa, transb, m, n, alpha_s, A_s, lda, beta_s, B_s, ldb, C_s, ldc);
  status = cublasDgeam_64(handle, transa, transb, m, n, alpha_d, A_d, lda, beta_d, B_d, ldb, C_d, ldc);
  status = cublasCgeam_64(handle, transa, transb, m, n, alpha_c, A_c, lda, beta_c, B_c, ldb, C_c, ldc);
  status = cublasZgeam_64(handle, transa, transb, m, n, alpha_z, A_z, lda, beta_z, B_z, ldb, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), side, m, n, A_s, lda, B_s, ldb, C_s, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), side, m, n, A_d, lda, B_d, ldb, C_d, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), side, m, n, (std::complex<float>*)A_c, lda, (std::complex<float>*)B_c, ldb, (std::complex<float>*)C_c, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), side, m, n, (std::complex<double>*)A_z, lda, (std::complex<double>*)B_z, ldb, (std::complex<double>*)C_z, ldc));
  status = cublasSdgmm_64(handle, side, m, n, A_s, lda, B_s, ldb, C_s, ldc);
  status = cublasDdgmm_64(handle, side, m, n, A_d, lda, B_d, ldb, C_d, ldc);
  status = cublasCdgmm_64(handle, side, m, n, A_c, lda, B_c, ldb, C_c, ldc);
  status = cublasZdgmm_64(handle, side, m, n, A_z, lda, B_z, ldb, C_z, ldc);
}
