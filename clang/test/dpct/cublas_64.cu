// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8
// RUN: dpct --format-range=none --usm-level=none --out-root %T/cublas_64 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/cublas_64/cublas_64.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cublas_64/cublas_64.dp.cpp -o %T/cublas_64/cublas_64.dp.o %}

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
  const half *alpha_h;
  const float *A_s;
  const double *A_d;
  const float2 *A_c;
  const double2 *A_z;
  const half *A_h;
  int64_t lda;
  const float *B_s;
  const double *B_d;
  const float2 *B_c;
  const double2 *B_z;
  const half *B_h;
  int64_t ldb;
  const float *beta_s;
  const double *beta_d;
  const float2 *beta_c;
  const double2 *beta_z;
  const half *beta_h;
  float *C_s;
  double *C_d;
  float2 *C_c;
  double2 *C_z;
  half *C_h;
  float *C1_s;
  double *C1_d;
  float2 *C1_c;
  double2 *C1_z;
  int64_t ldc;
  cublasFillMode_t uplo;
  cublasSideMode_t side;
  cublasDiagType_t diag;
  int64_t result;
  float result_s;
  double result_d;
  float2 result_c;
  double2 result_z;
  int64_t incx;
  int64_t incy;

  int64_t elemSize;
  cudaStream_t stream;
  //      CHECK: status = DPCT_CHECK_ERROR(dpct::blas::matrix_mem_copy(C_s, A_s, incy, incx, 1, n, elemSize));
  //      CHECK: status = DPCT_CHECK_ERROR(dpct::blas::matrix_mem_copy(C_s, A_s, incy, incx, 1, n, elemSize));
  //      CHECK: status = DPCT_CHECK_ERROR(dpct::blas::matrix_mem_copy(C_s, A_s, incy, incx, 1, n, elemSize, dpct::cs::memcpy_direction::automatic, *stream, true));
  //      CHECK: status = DPCT_CHECK_ERROR(dpct::blas::matrix_mem_copy(C_s, A_s, incy, incx, 1, n, elemSize, dpct::cs::memcpy_direction::automatic, *stream, true));
  status = cublasSetVector_64(n, elemSize, A_s, incx, C_s, incy);
  status = cublasGetVector_64(n, elemSize, A_s, incx, C_s, incy);
  status = cublasSetVectorAsync_64(n, elemSize, A_s, incx, C_s, incy, stream);
  status = cublasGetVectorAsync_64(n, elemSize, A_s, incx, C_s, incy, stream);

  //      CHECK: status = DPCT_CHECK_ERROR(dpct::blas::matrix_mem_copy(C_s, A_s, ldb, lda, m, n, elemSize));
  //      CHECK: status = DPCT_CHECK_ERROR(dpct::blas::matrix_mem_copy(C_s, A_s, ldb, lda, m, n, elemSize));
  //      CHECK: status = DPCT_CHECK_ERROR(dpct::blas::matrix_mem_copy(C_s, A_s, ldb, lda, m, n, elemSize, dpct::cs::memcpy_direction::automatic, *stream, true));
  //      CHECK: status = DPCT_CHECK_ERROR(dpct::blas::matrix_mem_copy(C_s, A_s, ldb, lda, m, n, elemSize, dpct::cs::memcpy_direction::automatic, *stream, true));
  status = cublasSetMatrix_64(m, n, elemSize, A_s, lda, C_s, ldb);
  status = cublasGetMatrix_64(m, n, elemSize, A_s, lda, C_s, ldb);
  status = cublasSetMatrixAsync_64(m, n, elemSize, A_s, lda, C_s, ldb, stream);
  status = cublasGetMatrixAsync_64(m, n, elemSize, A_s, lda, C_s, ldb, stream);

  //      CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasIsamax_64(handle, n, A_s, lda, &result);
  status = cublasIdamax_64(handle, n, A_d, lda, &result);
  status = cublasIcamax_64(handle, n, A_c, lda, &result);
  status = cublasIzamax_64(handle, n, A_z, lda, &result);

  //      CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), &result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasIsamin_64(handle, n, A_s, lda, &result);
  status = cublasIdamin_64(handle, n, A_d, lda, &result);
  status = cublasIcamin_64(handle, n, A_c, lda, &result);
  status = cublasIzamin_64(handle, n, A_z, lda, &result);

  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), &result_s);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), &result_d);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), &result_s);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), &result_d);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasSnrm2_64(handle, n, A_s, incx, &result_s);
  status = cublasDnrm2_64(handle, n, A_d, incx, &result_d);
  status = cublasScnrm2_64(handle, n, A_c, incx, &result_s);
  status = cublasDznrm2_64(handle, n, A_z, incx, &result_d);

  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float_out res_wrapper_ct6(handle->get_queue(), &result_s);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(B_s)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double_out res_wrapper_ct6(handle->get_queue(), &result_d);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(B_d)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float2_out res_wrapper_ct6(handle->get_queue(), &result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotu(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float2_out res_wrapper_ct6(handle->get_queue(), &result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotc(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double2_out res_wrapper_ct6(handle->get_queue(), &result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotu(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double2_out res_wrapper_ct6(handle->get_queue(), &result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotc(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasSdot_64(handle, n, A_s, incx, B_s, incy, &result_s);
  status = cublasDdot_64(handle, n, A_d, incx, B_d, incy, &result_d);
  status = cublasCdotu_64(handle, n, A_c, incx, B_c, incy, &result_c);
  status = cublasCdotc_64(handle, n, A_c, incx, B_c, incy, &result_c);
  status = cublasZdotu_64(handle, n, A_z, incx, B_z, incy, &result_z);
  status = cublasZdotc_64(handle, n, A_z, incx, B_z, incy, &result_z);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), incx));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), incx));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), incx));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), incx));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), incx));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), incx));
  status = cublasSscal_64(handle, n, alpha_s, C_s, incx);
  status = cublasDscal_64(handle, n, alpha_d, C_d, incx);
  status = cublasCscal_64(handle, n, alpha_c, C_c, incx);
  status = cublasZscal_64(handle, n, alpha_z, C_z, incx);
  status = cublasCsscal_64(handle, n, alpha_s, C_c, incx);
  status = cublasZdscal_64(handle, n, alpha_d, C_z, incx);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), incy));
  status = cublasSaxpy_64(handle, n, alpha_s, A_s, incx, C_s, incy);
  status = cublasDaxpy_64(handle, n, alpha_d, A_d, incx, C_d, incy);
  status = cublasCaxpy_64(handle, n, alpha_c, A_c, incx, C_c, incy);
  status = cublasZaxpy_64(handle, n, alpha_z, A_z, incx, C_z, incy);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), incy));
  status = cublasScopy_64(handle, n, A_s, incx, C_s, incy);
  status = cublasDcopy_64(handle, n, A_d, incx, C_d, incy);
  status = cublasCcopy_64(handle, n, A_c, incx, C_c, incy);
  status = cublasZcopy_64(handle, n, A_z, incx, C_z, incy);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C1_s)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C1_d)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C1_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C1_z)), incy));
  status = cublasSswap_64(handle, n, C_s, incx, C1_s, incy);
  status = cublasDswap_64(handle, n, C_d, incx, C1_d, incy);
  status = cublasCswap_64(handle, n, C_c, incx, C1_c, incy);
  status = cublasZswap_64(handle, n, C_z, incx, C1_z, incy);

  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), &result_s);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), &result_d);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), &result_s);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  //      CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), &result_d);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasSasum_64(handle, n, A_s, incx, &result_s);
  status = cublasDasum_64(handle, n, A_d, incx, &result_d);
  status = cublasScasum_64(handle, n, A_c, incx, &result_s);
  status = cublasDzasum_64(handle, n, A_z, incx, &result_d);

  const float *const_s;
  const double *const_d;
  const float2 *const_c;
  const double2 *const_z;
  float *s;
  double *d;
  float2 *c;
  double2 *z;

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C1_s)), incy, dpct::get_value(const_s, handle->get_queue()), dpct::get_value(const_s, handle->get_queue())));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C1_d)), incy, dpct::get_value(const_d, handle->get_queue()), dpct::get_value(const_d, handle->get_queue())));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C1_c)), incy, dpct::get_value(const_s, handle->get_queue()), dpct::get_value(const_c, handle->get_queue())));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C1_c)), incy, dpct::get_value(const_s, handle->get_queue()), dpct::get_value(const_s, handle->get_queue())));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C1_z)), incy, dpct::get_value(const_d, handle->get_queue()), dpct::get_value(const_z, handle->get_queue())));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C1_z)), incy, dpct::get_value(const_d, handle->get_queue()), dpct::get_value(const_d, handle->get_queue())));
  status = cublasSrot_64(handle, n, C_s, incx, C1_s, incy, const_s, const_s);
  status = cublasDrot_64(handle, n, C_d, incx, C1_d, incy, const_d, const_d);
  status = cublasCrot_64(handle, n, C_c, incx, C1_c, incy, const_s, const_c);
  status = cublasCsrot_64(handle, n, C_c, incx, C1_c, incy, const_s, const_s);
  status = cublasZrot_64(handle, n, C_z, incx, C1_z, incy, const_d, const_z);
  status = cublasZdrot_64(handle, n, C_z, incx, C1_z, incy, const_d, const_d);

  // CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float_in res_wrapper_ct6(handle->get_queue(), const_s, 5);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotm(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(s)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(s)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double_in res_wrapper_ct6(handle->get_queue(), const_d, 5);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotm(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(d)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(d)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasSrotm_64(handle, n, s, incx, s, incy, const_s);
  status = cublasDrotm_64(handle, n, d, incx, d, incy, const_d);

  const float *x_s;
  const double *x_d;
  const float2 *x_c;
  const double2 *x_z;
  float *y_s;
  double *y_d;
  float2 *y_c;
  double2 *y_z;
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemv(handle->get_queue(), transa, m, n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_s)), incx, dpct::get_value(beta_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemv(handle->get_queue(), transa, m, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_d)), incx, dpct::get_value(beta_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemv(handle->get_queue(), transa, m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemv(handle->get_queue(), transa, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  status = cublasSgemv_64(handle, transa, m, n, alpha_s, A_s, lda, x_s, incx, beta_s, y_s, incy);
  status = cublasDgemv_64(handle, transa, m, n, alpha_d, A_d, lda, x_d, incx, beta_d, y_d, incy);
  status = cublasCgemv_64(handle, transa, m, n, alpha_c, A_c, lda, x_c, incx, beta_c, y_c, incy);
  status = cublasZgemv_64(handle, transa, m, n, alpha_z, A_z, lda, x_z, incx, beta_z, y_z, incy);

  int64_t kl, ku;
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), transa, m, n, kl, ku, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_s)), incx, dpct::get_value(beta_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), transa, m, n, kl, ku, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_d)), incx, dpct::get_value(beta_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), transa, m, n, kl, ku, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), transa, m, n, kl, ku, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  status = cublasSgbmv_64(handle, transa, m, n, kl, ku, alpha_s, A_s, lda, x_s, incx, beta_s, y_s, incy);
  status = cublasDgbmv_64(handle, transa, m, n, kl, ku, alpha_d, A_d, lda, x_d, incx, beta_d, y_d, incy);
  status = cublasCgbmv_64(handle, transa, m, n, kl, ku, alpha_c, A_c, lda, x_c, incx, beta_c, y_c, incy);
  status = cublasZgbmv_64(handle, transa, m, n, kl, ku, alpha_z, A_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trmv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trmv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trmv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trmv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  status = cublasStrmv_64(handle, uplo, transa, diag, n, A_s, lda, y_s, incy);
  status = cublasDtrmv_64(handle, uplo, transa, diag, n, A_d, lda, y_d, incy);
  status = cublasCtrmv_64(handle, uplo, transa, diag, n, A_c, lda, y_c, incy);
  status = cublasZtrmv_64(handle, uplo, transa, diag, n, A_z, lda, y_z, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), uplo, transa, diag, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), uplo, transa, diag, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), uplo, transa, diag, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), uplo, transa, diag, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  status = cublasStbmv_64(handle, uplo, transa, diag, n, k, A_s, lda, y_s, incy);
  status = cublasDtbmv_64(handle, uplo, transa, diag, n, k, A_d, lda, y_d, incy);
  status = cublasCtbmv_64(handle, uplo, transa, diag, n, k, A_c, lda, y_c, incy);
  status = cublasZtbmv_64(handle, uplo, transa, diag, n, k, A_z, lda, y_z, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  status = cublasStpmv_64(handle, uplo, transa, diag, n, A_s, y_s, incy);
  status = cublasDtpmv_64(handle, uplo, transa, diag, n, A_d, y_d, incy);
  status = cublasCtpmv_64(handle, uplo, transa, diag, n, A_c, y_c, incy);
  status = cublasZtpmv_64(handle, uplo, transa, diag, n, A_z, y_z, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  status = cublasStrsv_64(handle, uplo, transa, diag, n, A_s, lda, y_s, incy);
  status = cublasDtrsv_64(handle, uplo, transa, diag, n, A_d, lda, y_d, incy);
  status = cublasCtrsv_64(handle, uplo, transa, diag, n, A_c, lda, y_c, incy);
  status = cublasZtrsv_64(handle, uplo, transa, diag, n, A_z, lda, y_z, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), uplo, transa, diag, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  status = cublasStpsv_64(handle, uplo, transa, diag, n, A_s, y_s, incy);
  status = cublasDtpsv_64(handle, uplo, transa, diag, n, A_d, y_d, incy);
  status = cublasCtpsv_64(handle, uplo, transa, diag, n, A_c, y_c, incy);
  status = cublasZtpsv_64(handle, uplo, transa, diag, n, A_z, y_z, incy);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(B_s)), ldb, dpct::get_value(beta_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(B_d)), ldb, dpct::get_value(beta_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha_h, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<sycl::half>(A_h)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<sycl::half>(B_h)), ldb, dpct::get_value(beta_h, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<sycl::half>(C_h)), ldc));
  status = cublasSgemm_64(handle, transa, transb, m, n, k, alpha_s, A_s, lda, B_s, ldb, beta_s, C_s, ldc);
  status = cublasDgemm_64(handle, transa, transb, m, n, k, alpha_d, A_d, lda, B_d, ldb, beta_d, C_d, ldc);
  status = cublasCgemm_64(handle, transa, transb, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZgemm_64(handle, transa, transb, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  status = cublasHgemm_64(handle, transa, transb, m, n, k, alpha_h, A_h, lda, B_h, ldb, beta_h, C_h, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc, oneapi::mkl::blas::compute_mode::complex_3m));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc, oneapi::mkl::blas::compute_mode::complex_3m));
  status = cublasCgemm3m_64(handle, transa, transb, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZgemm3m_64(handle, transa, transb, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  //      CHECK:   status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::get_value(beta_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), ldc));
  // CHECK-NEXT:   status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::get_value(beta_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), ldc));
  // CHECK-NEXT:   status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT:   status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  status = cublasSsyrk_64(handle, uplo, transa, n, k, alpha_s, A_s, lda, beta_s, C_s, ldc);
  status = cublasDsyrk_64(handle, uplo, transa, n, k, alpha_d, A_d, lda, beta_d, C_d, ldc);
  status = cublasCsyrk_64(handle, uplo, transa, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);
  status = cublasZsyrk_64(handle, uplo, transa, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(handle->get_queue(), side, uplo, m, n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(B_s)), ldb, dpct::get_value(beta_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(handle->get_queue(), side, uplo, m, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(B_d)), ldb, dpct::get_value(beta_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(handle->get_queue(), side, uplo, m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(handle->get_queue(), side, uplo, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  status = cublasSsymm_64(handle, side, uplo, m, n, alpha_s, A_s, lda, B_s, ldb, beta_s, C_s, ldc);
  status = cublasDsymm_64(handle, side, uplo, m, n, alpha_d, A_d, lda, B_d, ldb, beta_d, C_d, ldc);
  status = cublasCsymm_64(handle, side, uplo, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZsymm_64(handle, side, uplo, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(handle->get_queue(), side, uplo, transa, diag, m, n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(handle->get_queue(), side, uplo, transa, diag, m, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(handle->get_queue(), side, uplo, transa, diag, m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(handle->get_queue(), side, uplo, transa, diag, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  status = cublasStrsm_64(handle, side, uplo, transa, diag, m, n, alpha_s, A_s, lda, C_s, ldc);
  status = cublasDtrsm_64(handle, side, uplo, transa, diag, m, n, alpha_d, A_d, lda, C_d, ldc);
  status = cublasCtrsm_64(handle, side, uplo, transa, diag, m, n, alpha_c, A_c, lda, C_c, ldc);
  status = cublasZtrsm_64(handle, side, uplo, transa, diag, m, n, alpha_z, A_z, lda, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemm(handle->get_queue(), side, uplo, m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemm(handle->get_queue(), side, uplo, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  status = cublasChemm_64(handle, side, uplo, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZhemm_64(handle, side, uplo, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::herk(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::get_value(beta_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::herk(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::get_value(beta_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  status = cublasCherk_64(handle, uplo, transa, n, k, alpha_s, A_c, lda, beta_s, C_c, ldc);
  status = cublasZherk_64(handle, uplo, transa, n, k, alpha_d, A_z, lda, beta_d, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(B_s)), ldb, dpct::get_value(beta_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(B_d)), ldb, dpct::get_value(beta_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  status = cublasSsyr2k_64(handle, uplo, transa, n, k, alpha_s, A_s, lda, B_s, ldb, beta_s, C_s, ldc);
  status = cublasDsyr2k_64(handle, uplo, transa, n, k, alpha_d, A_d, lda, B_d, ldb, beta_d, C_d, ldc);
  status = cublasCsyr2k_64(handle, uplo, transa, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZsyr2k_64(handle, uplo, transa, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2k(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2k(handle->get_queue(), uplo, transa, n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  status = cublasCher2k_64(handle, uplo, transa, n, k, alpha_c, A_c, lda, B_c, ldb, beta_s, C_c, ldc);
  status = cublasZher2k_64(handle, uplo, transa, n, k, alpha_z, A_z, lda, B_z, ldb, beta_d, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::get_value(beta_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(B_s)), ldb, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::get_value(beta_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(B_d)), ldb, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  status = cublasSgeam_64(handle, transa, transb, m, n, alpha_s, A_s, lda, beta_s, B_s, ldb, C_s, ldc);
  status = cublasDgeam_64(handle, transa, transb, m, n, alpha_d, A_d, lda, beta_d, B_d, ldb, C_d, ldc);
  status = cublasCgeam_64(handle, transa, transb, m, n, alpha_c, A_c, lda, beta_c, B_c, ldb, C_c, ldc);
  status = cublasZgeam_64(handle, transa, transb, m, n, alpha_z, A_z, lda, beta_z, B_z, ldb, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), side, m, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(B_s)), ldb, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), side, m, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(B_d)), ldb, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), side, m, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), side, m, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  status = cublasSdgmm_64(handle, side, m, n, A_s, lda, B_s, ldb, C_s, ldc);
  status = cublasDdgmm_64(handle, side, m, n, A_d, lda, B_d, ldb, C_d, ldc);
  status = cublasCdgmm_64(handle, side, m, n, A_c, lda, B_c, ldb, C_c, ldc);
  status = cublasZdgmm_64(handle, side, m, n, A_z, lda, B_z, ldb, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(dpct::blas::trmm(handle, side, uplo, transa, diag, m, n, alpha_s, A_s, lda, B_s, ldb, C_s, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(dpct::blas::trmm(handle, side, uplo, transa, diag, m, n, alpha_d, A_d, lda, B_d, ldb, C_d, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(dpct::blas::trmm(handle, side, uplo, transa, diag, m, n, alpha_c, A_c, lda, B_c, ldb, C_c, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(dpct::blas::trmm(handle, side, uplo, transa, diag, m, n, alpha_z, A_z, lda, B_z, ldb, C_z, ldc));
  status = cublasStrmm_64(handle, side, uplo, transa, diag, m, n, alpha_s, A_s, lda, B_s, ldb, C_s, ldc);
  status = cublasDtrmm_64(handle, side, uplo, transa, diag, m, n, alpha_d, A_d, lda, B_d, ldb, C_d, ldc);
  status = cublasCtrmm_64(handle, side, uplo, transa, diag, m, n, alpha_c, A_c, lda, B_c, ldb, C_c, ldc);
  status = cublasZtrmm_64(handle, side, uplo, transa, diag, m, n, alpha_z, A_z, lda, B_z, ldb, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(dpct::blas::syrk(handle, uplo, transa, n, k, alpha_s, A_s, lda, B_s, ldb, beta_s, C_s, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(dpct::blas::syrk(handle, uplo, transa, n, k, alpha_d, A_d, lda, B_d, ldb, beta_d, C_d, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(dpct::blas::syrk(handle, uplo, transa, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(dpct::blas::syrk(handle, uplo, transa, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc));
  status = cublasSsyrkx_64(handle, uplo, transa, n, k, alpha_s, A_s, lda, B_s, ldb, beta_s, C_s, ldc);
  status = cublasDsyrkx_64(handle, uplo, transa, n, k, alpha_d, A_d, lda, B_d, ldb, beta_d, C_d, ldc);
  status = cublasCsyrkx_64(handle, uplo, transa, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZsyrkx_64(handle, uplo, transa, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(dpct::blas::herk(handle, uplo, transa, n, k, alpha_c, A_c, lda, B_c, ldb, beta_s, C_c, ldc));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(dpct::blas::herk(handle, uplo, transa, n, k, alpha_z, A_z, lda, B_z, ldb, beta_d, C_z, ldc));
  status = cublasCherkx_64(handle, uplo, transa, n, k, alpha_c, A_c, lda, B_c, ldb, beta_s, C_c, ldc);
  status = cublasZherkx_64(handle, uplo, transa, n, k, alpha_z, A_z, lda, B_z, ldb, beta_d, C_z, ldc);

  cudaDataType type_x;
  cudaDataType type_y;
  cudaDataType type_res;
  cudaDataType type_exec;
  cudaDataType type_alpha;
  cudaDataType type_cs;
  void *res;
  void *x;
  void *y;
  void *alpha;
  //      CHECK: status = DPCT_CHECK_ERROR(dpct::blas::nrm2(handle, n, x, type_x, incx, res, type_res));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(dpct::blas::dot(handle, n, x, type_x, incx, y, type_y, incy, res, type_res));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(dpct::blas::dotc(handle, n, x, type_x, incx, y, type_y, incy, res, type_res));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(dpct::blas::scal(handle, n, alpha, type_alpha, x, type_x, incx));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(dpct::blas::axpy(handle, n, alpha, type_alpha, x, type_x, incx, y, type_y, incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(dpct::blas::rot(handle, n, x, type_x, incx, y, type_y, incy, c, s, type_cs));
  status = cublasNrm2Ex_64(handle, n, x, type_x, incx, res, type_res, type_exec);
  status = cublasDotEx_64(handle, n, x, type_x, incx, y, type_y, incy, res, type_res, type_exec);
  status = cublasDotcEx_64(handle, n, x, type_x, incx, y, type_y, incy, res, type_res, type_exec);
  status = cublasScalEx_64(handle, n, alpha, type_alpha, x, type_x, incx, type_exec);
  status = cublasAxpyEx_64(handle, n, alpha, type_alpha, x, type_x, incx, y, type_y, incy, type_exec);
  status = cublasRotEx_64(handle, n, x, type_x, incx, y, type_y, incy, c, s, type_cs, type_exec);

  void **a_array;
  void **b_array;
  void **c_array;
  void *aa;
  void *bb;
  void *cc;
  cudaDataType type_a;
  cudaDataType type_b;
  cudaDataType type_c;
  void *beta;
  cublasGemmAlgo_t algo;
  int64_t batch;
  int64_t stride_a;
  int64_t stride_b;
  int64_t stride_c;
  cublasComputeType_t type_compute;
  // CHECK: status = DPCT_CHECK_ERROR(dpct::blas::gemm_batch(handle, transa, transb, m, n, k, alpha, aa, type_a, lda, stride_a, bb, type_b, ldb, stride_b, beta, cc, type_c, ldc, stride_c, batch, type_compute));
  status = cublasGemmStridedBatchedEx_64(handle, transa, transb, m, n, k, alpha, aa, type_a, lda, stride_a, bb, type_b, ldb, stride_b, beta, cc, type_c, ldc, stride_c, batch, type_compute, algo);

  // CHECK: dpct::blas::gemm(handle, transa, transb, m, n, k, alpha_s, A_s, type_a, lda, B_s, type_b, ldb, beta_s, C_s, type_c, ldc, dpct::library_data_t::real_float);
  // CHECK-NEXT: dpct::blas::gemm(handle, transa, transb, m, n, k, alpha_c, A_c, type_a, lda, B_c, type_b, ldb, beta_c, C_c, type_c, ldc, dpct::library_data_t::complex_float);
  // CHECK-NEXT: dpct::blas::gemm(handle, transa, transb, m, n, k, alpha_c, A_c, type_a, lda, B_c, type_b, ldb, beta_c, C_c, type_c, ldc, dpct::library_data_t::complex_float);
  // CHECK-NEXT: dpct::blas::gemm(handle, transa, transb, m, n, k, alpha_c, A_c, type_a, lda, B_c, type_b, ldb, beta_c, C_c, type_c, ldc, type_compute);
  cublasSgemmEx_64(handle, transa, transb, m, n, k, alpha_s, A_s, type_a, lda, B_s, type_b, ldb, beta_s, C_s, type_c, ldc);
  cublasCgemmEx_64(handle, transa, transb, m, n, k, alpha_c, A_c, type_a, lda, B_c, type_b, ldb, beta_c, C_c, type_c, ldc);
  cublasCgemm3mEx_64(handle, transa, transb, m, n, k, alpha_c, A_c, type_a, lda, B_c, type_b, ldb, beta_c, C_c, type_c, ldc);
  cublasGemmEx_64(handle, transa, transb, m, n, k, alpha_c, A_c, type_a, lda, B_c, type_b, ldb, beta_c, C_c, type_c, ldc, type_compute, algo);

  cublasOperation_t trans;
  // CHECK: dpct::blas::syherk<false>(handle, uplo, trans, n, k, alpha_c, A_c, type_a, lda, beta_c, C_c, type_c, ldc, dpct::library_data_t::complex_float);
  // CHECK-NEXT: dpct::blas::syherk<false>(handle, uplo, trans, n, k, alpha_c, A_c, type_a, lda, beta_c, C_c, type_c, ldc, dpct::library_data_t::complex_float);
  // CHECK-NEXT: dpct::blas::syherk<true>(handle, uplo, trans, n, k, alpha_s, A_c, type_a, lda, beta_s, C_c, type_c, ldc, dpct::library_data_t::complex_float);
  // CHECK-NEXT: dpct::blas::syherk<true>(handle, uplo, trans, n, k, alpha_s, A_c, type_a, lda, beta_s, C_c, type_c, ldc, dpct::library_data_t::complex_float);
  cublasCsyrkEx_64(handle, uplo, trans, n, k, alpha_c, A_c, type_a, lda, beta_c, C_c, type_c, ldc);
  cublasCsyrk3mEx_64(handle, uplo, trans, n, k, alpha_c, A_c, type_a, lda, beta_c, C_c, type_c, ldc);
  cublasCherkEx_64(handle, uplo, trans, n, k, alpha_s, A_c, type_a, lda, beta_s, C_c, type_c, ldc);
  cublasCherk3mEx_64(handle, uplo, trans, n, k, alpha_s, A_c, type_a, lda, beta_s, C_c, type_c, ldc);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), uplo, transa, diag, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incx));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), uplo, transa, diag, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incx));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), uplo, transa, diag, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incx));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), uplo, transa, diag, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incx));
  status = cublasStbsv_64(handle, uplo, transa, diag, n, k, A_s, lda, y_s, incx);
  status = cublasDtbsv_64(handle, uplo, transa, diag, n, k, A_d, lda, y_d, incx);
  status = cublasCtbsv_64(handle, uplo, transa, diag, n, k, A_c, lda, y_c, incx);
  status = cublasZtbsv_64(handle, uplo, transa, diag, n, k, A_z, lda, y_z, incx);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(handle->get_queue(), uplo, n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_s)), incx, dpct::get_value(beta_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(handle->get_queue(), uplo, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_d)), incx, dpct::get_value(beta_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(handle->get_queue(), uplo, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(handle->get_queue(), uplo, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  status = cublasSsymv_64(handle, uplo, n, alpha_s, A_s, lda, x_s, incx, beta_s, y_s, incy);
  status = cublasDsymv_64(handle, uplo, n, alpha_d, A_d, lda, x_d, incx, beta_d, y_d, incy);
  status = cublasCsymv_64(handle, uplo, n, alpha_c, A_c, lda, x_c, incx, beta_c, y_c, incy);
  status = cublasZsymv_64(handle, uplo, n, alpha_z, A_z, lda, x_z, incx, beta_z, y_z, incy);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemv(handle->get_queue(), uplo, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemv(handle->get_queue(), uplo, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  status = cublasChemv_64(handle, uplo, n, alpha_c, A_c, lda, x_c, incx, beta_c, y_c, incy);
  status = cublasZhemv_64(handle, uplo, n, alpha_z, A_z, lda, x_z, incx, beta_z, y_z, incy);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::sbmv(handle->get_queue(), uplo, n, k, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_s)), incx, dpct::get_value(beta_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::sbmv(handle->get_queue(), uplo, n, k, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_d)), incx, dpct::get_value(beta_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incy));
  status = cublasSsbmv_64(handle, uplo, n, k, alpha_s, A_s, lda, x_s, incx, beta_s, y_s, incy);
  status = cublasDsbmv_64(handle, uplo, n, k, alpha_d, A_d, lda, x_d, incx, beta_d, y_d, incy);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hbmv(handle->get_queue(), uplo, n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hbmv(handle->get_queue(), uplo, n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  status = cublasChbmv_64(handle, uplo, n, k, alpha_c, A_c, lda, x_c, incx, beta_c, y_c, incy);
  status = cublasZhbmv_64(handle, uplo, n, k, alpha_z, A_z, lda, x_z, incx, beta_z, y_z, incy);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::spmv(handle->get_queue(), uplo, n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_s)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_s)), incx, dpct::get_value(beta_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::spmv(handle->get_queue(), uplo, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_d)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_d)), incx, dpct::get_value(beta_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incy));
  status = cublasSspmv_64(handle, uplo, n, alpha_s, A_s, x_s, incx, beta_s, y_s, incy);
  status = cublasDspmv_64(handle, uplo, n, alpha_d, A_d, x_d, incx, beta_d, y_d, incy);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpmv(handle->get_queue(), uplo, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpmv(handle->get_queue(), uplo, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  status = cublasChpmv_64(handle, uplo, n, alpha_c, A_c, x_c, incx, beta_c, y_c, incy);
  status = cublasZhpmv_64(handle, uplo, n, alpha_z, A_z, x_z, incx, beta_z, y_z, incy);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::ger(handle->get_queue(), m, n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_s)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), lda));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::ger(handle->get_queue(), m, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_d)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), lda));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::geru(handle->get_queue(), m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), lda));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gerc(handle->get_queue(), m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), lda));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::geru(handle->get_queue(), m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), lda));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gerc(handle->get_queue(), m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), lda));
  status = cublasSger_64(handle, m, n, alpha_s, x_s, incx, y_s, incy, C_s, lda);
  status = cublasDger_64(handle, m, n, alpha_d, x_d, incx, y_d, incy, C_d, lda);
  status = cublasCgeru_64(handle, m, n, alpha_c, x_c, incx, y_c, incy, C_c, lda);
  status = cublasCgerc_64(handle, m, n, alpha_c, x_c, incx, y_c, incy, C_c, lda);
  status = cublasZgeru_64(handle, m, n, alpha_z, x_z, incx, y_z, incy, C_z, lda);
  status = cublasZgerc_64(handle, m, n, alpha_z, x_z, incx, y_z, incy, C_z, lda);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr(handle->get_queue(), uplo, n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_s)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), lda));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr(handle->get_queue(), uplo, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_d)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), lda));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr(handle->get_queue(), uplo, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), lda));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr(handle->get_queue(), uplo, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), lda));
  status = cublasSsyr_64(handle, uplo, n, alpha_s, x_s, incx, C_s, lda);
  status = cublasDsyr_64(handle, uplo, n, alpha_d, x_d, incx, C_d, lda);
  status = cublasCsyr_64(handle, uplo, n, alpha_c, x_c, incx, C_c, lda);
  status = cublasZsyr_64(handle, uplo, n, alpha_z, x_z, incx, C_z, lda);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her(handle->get_queue(), uplo, n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), lda));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her(handle->get_queue(), uplo, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), lda));
  status = cublasCher_64(handle, uplo, n, alpha_s, x_c, incx, C_c, lda);
  status = cublasZher_64(handle, uplo, n, alpha_d, x_z, incx, C_z, lda);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::spr(handle->get_queue(), uplo, n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_s)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s))));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::spr(handle->get_queue(), uplo, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_d)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d))));
  status = cublasSspr_64(handle, uplo, n, alpha_s, x_s, incx, C_s);
  status = cublasDspr_64(handle, uplo, n, alpha_d, x_d, incx, C_d);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpr(handle->get_queue(), uplo, n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c))));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpr(handle->get_queue(), uplo, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z))));
  status = cublasChpr_64(handle, uplo, n, alpha_s, x_c, incx, C_c);
  status = cublasZhpr_64(handle, uplo, n, alpha_d, x_z, incx, C_z);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2(handle->get_queue(), uplo, n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_s)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s)), lda));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2(handle->get_queue(), uplo, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_d)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d)), lda));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2(handle->get_queue(), uplo, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), lda));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2(handle->get_queue(), uplo, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), lda));
  status = cublasSsyr2_64(handle, uplo, n, alpha_s, x_s, incx, y_s, incy, C_s, lda);
  status = cublasDsyr2_64(handle, uplo, n, alpha_d, x_d, incx, y_d, incy, C_d, lda);
  status = cublasCsyr2_64(handle, uplo, n, alpha_c, x_c, incx, y_c, incy, C_c, lda);
  status = cublasZsyr2_64(handle, uplo, n, alpha_z, x_z, incx, y_z, incy, C_z, lda);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2(handle->get_queue(), uplo, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), lda));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2(handle->get_queue(), uplo, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), lda));
  status = cublasCher2_64(handle, uplo, n, alpha_c, x_c, incx, y_c, incy, C_c, lda);
  status = cublasZher2_64(handle, uplo, n, alpha_z, x_z, incx, y_z, incy, C_z, lda);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::spr2(handle->get_queue(), uplo, n, dpct::get_value(alpha_s, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_s)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y_s)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_s))));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::spr2(handle->get_queue(), uplo, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_d)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(y_d)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_d))));
  status = cublasSspr2_64(handle, uplo, n, alpha_s, x_s, incx, y_s, incy, C_s);
  status = cublasDspr2_64(handle, uplo, n, alpha_d, x_d, incx, y_d, incy, C_d);

  //      CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpr2(handle->get_queue(), uplo, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c))));
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpr2(handle->get_queue(), uplo, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z))));
  status = cublasChpr2_64(handle, uplo, n, alpha_c, x_c, incx, y_c, incy, C_c);
  status = cublasZhpr2_64(handle, uplo, n, alpha_z, x_z, incx, y_z, incy, C_z);
}

void foo2() {
  cublasHandle_t handle;
  int n;
  void *x, *y;
  int incx, incy;
  void *res;
  int64_t *idx;
  void *param;
  // CHECK: dpct::blas::copy(handle, n, x, dpct::library_data_t::real_float, incx, y, dpct::library_data_t::real_float, incy);
  // CHECK-NEXT: dpct::blas::swap(handle, n, x, dpct::library_data_t::real_float, incx, y, dpct::library_data_t::real_float, incy);
  // CHECK-NEXT: dpct::blas::iamax(handle, n, x, dpct::library_data_t::real_float, incx, idx);
  // CHECK-NEXT: dpct::blas::iamin(handle, n, x, dpct::library_data_t::real_float, incx, idx);
  // CHECK-NEXT: dpct::blas::asum(handle, n, x, dpct::library_data_t::real_float, incx, res, dpct::library_data_t::real_float);
  // CHECK-NEXT: dpct::blas::rotm(handle, n, x, dpct::library_data_t::real_float, incx, y, dpct::library_data_t::real_float, incy, param, dpct::library_data_t::real_float);
  cublasCopyEx_64(handle, n, x, CUDA_R_32F, incx, y, CUDA_R_32F, incy);
  cublasSwapEx_64(handle, n, x, CUDA_R_32F, incx, y, CUDA_R_32F, incy);
  cublasIamaxEx_64(handle, n, x, CUDA_R_32F, incx, idx);
  cublasIaminEx_64(handle, n, x, CUDA_R_32F, incx, idx);
  cublasAsumEx_64(handle, n, x, CUDA_R_32F, incx, res, CUDA_R_32F, CUDA_R_32F);
  cublasRotmEx_64(handle, n, x, CUDA_R_32F, incx, y, CUDA_R_32F, incy, param, CUDA_R_32F, CUDA_R_32F);
}
