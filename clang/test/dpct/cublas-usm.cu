// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-usm.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>


int main() {
  cublasHandle_t handle;
  int N = 275;
  float *h_a, *h_b, *h_c;
  const float *d_A_S;
  const float *d_B_S;
  float *d_C_S;
  float alpha_S = 1.0f;
  float beta_S = 0.0f;
  int trans0 = 0;
  int trans1 = 1;
  int trans2 = 2;
  int fill0 = 0;
  int side0 = 0;
  int diag0 = 0;
  int *result = 0;
  const float *x_S = 0;
  const float *y_S = 0;

  const double *d_A_D;
  const double  *d_B_D;
  double  *d_C_D;
  double alpha_D;
  double beta_D;
  const double *x_D;
  const double *y_D;

  const float2 *d_A_C;
  const float2  *d_B_C;
  float2  *d_C_C;
  float2 alpha_C;
  float2 beta_C;
  const float2 *x_C;
  const float2 *y_C;

  const double2 *d_A_Z;
  const double2  *d_B_Z;
  double2  *d_C_Z;
  double2 alpha_Z;
  double2 beta_Z;
  const double2 *x_Z;
  const double2 *y_Z;

  float* result_S;
  double* result_D;
  float2* result_C;
  double2* result_Z;

  int incx, incy, lda, ldb, ldc;

  //CHECK:/*
  //CHECK-NEXT:DPCT1018:{{[0-9]+}}: The cublasSetVector was migrated, but due to parameter 11111 equals to parameter 11111 but greater than 1, the generated code performance may be sub-optimal.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int a = (dpct::get_default_queue().memcpy(d_C_S, h_a, (10)*(sizeof(float))*(11111)).wait(), 0);
  //CHECK-NEXT:dpct::get_default_queue().memcpy(d_C_S, h_b, (10)*(sizeof(float))*(1)).wait();
  //CHECK-NEXT:dpct::get_default_queue().memcpy(d_C_S, h_c, (10)*(sizeof(float))*(1)).wait();
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (dpct::get_default_queue().memcpy(d_C_S, h_a, (100)*(100)*(10000)).wait(), 0);
  int a = cublasSetVector(10, sizeof(float), h_a, 11111, d_C_S, 11111);
  cublasSetVector(10, sizeof(float), h_b, 1, d_C_S, 1);
  cublasSetVector(10, sizeof(float), h_c, 1, d_C_S, 1);
  a = cublasSetMatrix(100, 100, 10000, h_a, 100, d_C_S, 100);


  //level 1

  //CHECK:{
  //CHECK-NEXT:int64_t result_temp_value;
  //CHECK-NEXT:int64_t* result_temp_ptr = (int64_t*)sycl::malloc_device(sizeof(int64_t), dpct::get_current_device(), dpct::get_default_context());
  //CHECK-NEXT:a = (mkl::blas::iamax(handle, N, x_S, N, result_temp_ptr).wait(), 0);
  //CHECK-NEXT:dpct::get_default_queue().memcpy(&result_temp_value, result_temp_ptr, sizeof(int64_t)).wait();
  //CHECK-NEXT:*(result) = (int)result_temp_value;
  //CHECK-NEXT:}
  a = cublasIsamax(handle, N, x_S, N, result);
  //CHECK:{
  //CHECK-NEXT:int64_t result_temp_value;
  //CHECK-NEXT:int64_t* result_temp_ptr = (int64_t*)sycl::malloc_device(sizeof(int64_t), dpct::get_current_device(), dpct::get_default_context());
  //CHECK-NEXT:mkl::blas::iamax(handle, N, x_D, N, result_temp_ptr).wait();
  //CHECK-NEXT:dpct::get_default_queue().memcpy(&result_temp_value, result_temp_ptr, sizeof(int64_t)).wait();
  //CHECK-NEXT:*(result) = (int)result_temp_value;
  //CHECK-NEXT:}
  cublasIdamax(handle, N, x_D, N, result);
  //CHECK:{
  //CHECK-NEXT:int64_t result_temp_value;
  //CHECK-NEXT:int64_t* result_temp_ptr = (int64_t*)sycl::malloc_device(sizeof(int64_t), dpct::get_current_device(), dpct::get_default_context());
  //CHECK-NEXT:a = (mkl::blas::iamax(handle, N, (std::complex<float>*)(x_C), N, result_temp_ptr).wait(), 0);
  //CHECK-NEXT:dpct::get_default_queue().memcpy(&result_temp_value, result_temp_ptr, sizeof(int64_t)).wait();
  //CHECK-NEXT:*(result) = (int)result_temp_value;
  //CHECK-NEXT:}
  a = cublasIcamax(handle, N, x_C, N, result);
  //CHECK:{
  //CHECK-NEXT:int64_t result_temp_value;
  //CHECK-NEXT:int64_t* result_temp_ptr = (int64_t*)sycl::malloc_device(sizeof(int64_t), dpct::get_current_device(), dpct::get_default_context());
  //CHECK-NEXT:mkl::blas::iamax(handle, N, (std::complex<double>*)(x_Z), N, result_temp_ptr).wait();
  //CHECK-NEXT:dpct::get_default_queue().memcpy(&result_temp_value, result_temp_ptr, sizeof(int64_t)).wait();
  //CHECK-NEXT:*(result) = (int)result_temp_value;
  //CHECK-NEXT:}
  cublasIzamax(handle, N, x_Z, N, result);

  //CHECK:a = (mkl::blas::rotm(handle, N, d_C_S, N, d_C_S, N, const_cast<float*>(x_S)).wait(), 0);
  a = cublasSrotm(handle, N, d_C_S, N, d_C_S, N, x_S);
  //CHECK:mkl::blas::rotm(handle, N, d_C_D, N, d_C_D, N, const_cast<double*>(x_D)).wait();
  cublasDrotm(handle, N, d_C_D, N, d_C_D, N, x_D);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::copy(handle, N, x_S, incx, d_C_S, incy).wait(), 0);
  a = cublasScopy(handle, N, x_S, incx, d_C_S, incy);
  // CHECK:mkl::blas::copy(handle, N, x_D, incx, d_C_D, incy).wait();
  cublasDcopy(handle, N, x_D, incx, d_C_D, incy);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::copy(handle, N, (std::complex<float>*)(x_C), incx, (std::complex<float>*)(d_C_C), incy).wait(), 0);
  a = cublasCcopy(handle, N, x_C, incx, d_C_C, incy);
  // CHECK:mkl::blas::copy(handle, N, (std::complex<double>*)(x_Z), incx, (std::complex<double>*)(d_C_Z), incy).wait();
  cublasZcopy(handle, N, x_Z, incx, d_C_Z, incy);


  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::axpy(handle, N, alpha_S, x_S, incx, result_S, incy).wait(), 0);
  a = cublasSaxpy(handle, N, &alpha_S, x_S, incx, result_S, incy);
  // CHECK:mkl::blas::axpy(handle, N, alpha_D, x_D, incx, result_D, incy).wait();
  cublasDaxpy(handle, N, &alpha_D, x_D, incx, result_D, incy);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::axpy(handle, N, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)(x_C), incx, (std::complex<float>*)(result_C), incy).wait(), 0);
  a = cublasCaxpy(handle, N, &alpha_C, x_C, incx, result_C, incy);
  // CHECK:mkl::blas::axpy(handle, N, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)(x_Z), incx, (std::complex<double>*)(result_Z), incy).wait();
  cublasZaxpy(handle, N, &alpha_Z, x_Z, incx, result_Z, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::scal(handle, N, alpha_S, result_S, incx).wait(), 0);
  a = cublasSscal(handle, N, &alpha_S, result_S, incx);
  // CHECK:mkl::blas::scal(handle, N, alpha_D, result_D, incx).wait();
  cublasDscal(handle, N, &alpha_D, result_D, incx);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::scal(handle, N, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)(result_C), incx).wait(), 0);
  a = cublasCscal(handle, N, &alpha_C, result_C, incx);
  // CHECK:mkl::blas::scal(handle, N, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)(result_Z), incx).wait();
  cublasZscal(handle, N, &alpha_Z, result_Z, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::nrm2(handle, N, x_S, incx, result_S).wait(), 0);
  a = cublasSnrm2(handle, N, x_S, incx, result_S);
  // CHECK:mkl::blas::nrm2(handle, N, x_D, incx, result_D).wait();

  cublasDnrm2(handle, N, x_D, incx, result_D);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::nrm2(handle, N, (std::complex<float>*)(x_C), incx, (float*)(result_S)).wait(), 0);
  a = cublasScnrm2(handle, N, x_C, incx, result_S);
  // CHECK:mkl::blas::nrm2(handle, N, (std::complex<double>*)(x_Z), incx, (double*)(result_D)).wait();
  cublasDznrm2(handle, N, x_Z, incx, result_D);


  //level 2

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct1 = trans2;
  // CHECK-NEXT: a = (mkl::blas::gemv(handle, (int)transpose_ct1==2 ? mkl::transpose::conjtrans : (mkl::transpose)transpose_ct1, N, N, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy).wait(), 0);
  // CHECK-NEXT: }
  a = cublasSgemv(handle, (cublasOperation_t)trans2, N, N, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  // CHECK:mkl::blas::gemv(handle, mkl::transpose::nontrans, N, N, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy).wait();
  cublasDgemv(handle, CUBLAS_OP_N, N, N, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct1 = trans2;
  // CHECK-NEXT: a = (mkl::blas::gemv(handle, (int)transpose_ct1==2 ? mkl::transpose::conjtrans : (mkl::transpose)transpose_ct1, N, N, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)(x_C), lda, (std::complex<float>*)(y_C), incx, std::complex<float>(beta_C.x(),beta_C.y()), (std::complex<float>*)(result_C), incy).wait(), 0);
  // CHECK-NEXT: }
  a = cublasCgemv(handle, (cublasOperation_t)trans2, N, N, &alpha_C, x_C, lda, y_C, incx, &beta_C, result_C, incy);
  // CHECK:mkl::blas::gemv(handle, mkl::transpose::nontrans, N, N, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)(x_Z), lda, (std::complex<double>*)(y_Z), incx, std::complex<double>(beta_Z.x(),beta_Z.y()), (std::complex<double>*)(result_Z), incy).wait();
  cublasZgemv(handle, CUBLAS_OP_N, N, N, &alpha_Z, x_Z, lda, y_Z, incx, &beta_Z, result_Z, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::ger(handle, N, N, alpha_S, x_S, incx, y_S, incy, result_S, lda).wait(), 0);
  a = cublasSger(handle, N, N, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  // CHECK:mkl::blas::ger(handle, N, N, alpha_D, x_D, incx, y_D, incy, result_D, lda).wait();
  cublasDger(handle, N, N, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::geru(handle, N, N, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)(x_C), incx, (std::complex<float>*)(y_C), incy, (std::complex<float>*)(result_C), lda).wait(), 0);
  a = cublasCgeru(handle, N, N, &alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK:mkl::blas::gerc(handle, N, N, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)(x_C), incx, (std::complex<float>*)(y_C), incy, (std::complex<float>*)(result_C), lda).wait();
  cublasCgerc(handle, N, N, &alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::geru(handle, N, N, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)(x_Z), incx, (std::complex<double>*)(y_Z), incy, (std::complex<double>*)(result_Z), lda).wait(), 0);
  a = cublasZgeru(handle, N, N, &alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);
  // CHECK:mkl::blas::gerc(handle, N, N, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)(x_Z), incx, (std::complex<double>*)(y_Z), incy, (std::complex<double>*)(result_Z), lda).wait();
  cublasZgerc(handle, N, N, &alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);








  //level 3

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:{
  //CHECK-NEXT:auto transpose_ct1 = trans0;
  //CHECK-NEXT:auto transpose_ct2 = trans1;
  //CHECK-NEXT:a = (mkl::blas::gemm(handle, (int)transpose_ct1==2 ? mkl::transpose::conjtrans : (mkl::transpose)transpose_ct1, (int)transpose_ct2==2 ? mkl::transpose::conjtrans : (mkl::transpose)transpose_ct2, N, N, N, alpha_S, d_A_S, N, d_B_S, N, beta_S, d_C_S, N).wait(), 0);
  //CHECK-NEXT:}
  a = cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  //CHECK:{
  //CHECK-NEXT:auto transpose_ct1 = trans0;
  //CHECK-NEXT:auto transpose_ct2 = trans1;
  //CHECK-NEXT:mkl::blas::gemm(handle, (int)transpose_ct1==2 ? mkl::transpose::conjtrans : (mkl::transpose)transpose_ct1, (int)transpose_ct2==2 ? mkl::transpose::conjtrans : (mkl::transpose)transpose_ct2, N, N, N, alpha_D, d_A_D, N, d_B_D, N, beta_D, d_C_D, N).wait();
  //CHECK-NEXT:}
  cublasDgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)(d_A_C), N, (std::complex<float>*)(d_B_C), N, std::complex<float>(beta_C.x(),beta_C.y()), (std::complex<float>*)(d_C_C), N).wait(), 0);
  a = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N);
  //CHECK:mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)(d_A_Z), N, (std::complex<double>*)(d_B_Z), N, std::complex<double>(beta_Z.x(),beta_Z.y()), (std::complex<double>*)(d_C_Z), N).wait();
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, &beta_Z, d_C_Z, N);


  //CHECK:{
  //CHECK-NEXT:auto transpose_ct3 = trans0;
  //CHECK-NEXT:auto ptr_ct8 = d_A_S;
  //CHECK-NEXT:auto ptr_ct12 = d_C_S;
  //CHECK-NEXT:auto ld_ct13 = N; auto m_ct5 = N; auto n_ct6 = N;
  //CHECK-NEXT:dpct::matrix_mem_copy(ptr_ct12, d_B_S, ld_ct13, N, m_ct5, n_ct6, dpct::device_to_device, handle);
  //CHECK-NEXT:a = (mkl::blas::trmm(handle, (mkl::side)side0, (int)fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, (int)transpose_ct3==2 ? mkl::transpose::conjtrans : (mkl::transpose)transpose_ct3, (mkl::diag)diag0, m_ct5, n_ct6, alpha_S, ptr_ct8, N,  ptr_ct12, ld_ct13).wait(), 0);
  //CHECK-NEXT:}
  a = cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N);
  //CHECK:{
  //CHECK-NEXT:auto transpose_ct3 = trans0;
  //CHECK-NEXT:auto ptr_ct8 = d_A_D;
  //CHECK-NEXT:auto ptr_ct12 = d_C_D;
  //CHECK-NEXT:auto ld_ct13 = N; auto m_ct5 = N; auto n_ct6 = N;
  //CHECK-NEXT:dpct::matrix_mem_copy(ptr_ct12, d_B_D, ld_ct13, N, m_ct5, n_ct6, dpct::device_to_device, handle);
  //CHECK-NEXT:mkl::blas::trmm(handle, (mkl::side)side0, (int)fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, (int)transpose_ct3==2 ? mkl::transpose::conjtrans : (mkl::transpose)transpose_ct3, (mkl::diag)diag0, m_ct5, n_ct6, alpha_D, ptr_ct8, N,  ptr_ct12, ld_ct13).wait();
  //CHECK-NEXT:}
  cublasDtrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_D, d_A_D, N, d_B_D, N, d_C_D, N);
  //CHECK:{
  //CHECK-NEXT:auto ptr_ct8 = d_A_C;
  //CHECK-NEXT:auto ptr_ct12 = d_C_C;
  //CHECK-NEXT:auto ld_ct13 = N; auto m_ct5 = N; auto n_ct6 = N;
  //CHECK-NEXT:dpct::matrix_mem_copy(ptr_ct12, d_B_C, ld_ct13, N, m_ct5, n_ct6, dpct::device_to_device, handle);
  //CHECK-NEXT:a = (mkl::blas::trmm(handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, m_ct5, n_ct6, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)(ptr_ct8), N,  (std::complex<float>*)(ptr_ct12), ld_ct13).wait(), 0);
  //CHECK-NEXT:}
  a = cublasCtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, N, &alpha_C, d_A_C, N, d_B_C, N, d_C_C, N);
  //CHECK:{
  //CHECK-NEXT:auto ptr_ct8 = d_A_Z;
  //CHECK-NEXT:auto ptr_ct12 = d_C_Z;
  //CHECK-NEXT:auto ld_ct13 = N; auto m_ct5 = N; auto n_ct6 = N;
  //CHECK-NEXT:dpct::matrix_mem_copy(ptr_ct12, d_B_Z, ld_ct13, N, m_ct5, n_ct6, dpct::device_to_device, handle);
  //CHECK-NEXT:mkl::blas::trmm(handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, m_ct5, n_ct6, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)(ptr_ct8), N,  (std::complex<double>*)(ptr_ct12), ld_ct13).wait();
  //CHECK-NEXT:}
  cublasZtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, d_C_Z, N);


  //CHECK:{
  //CHECK-NEXT:auto transpose_ct2 = trans1;
  //CHECK-NEXT:a = (mkl::blas::gemmt(handle, (int)fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, (int)transpose_ct2==2 ? mkl::transpose::conjtrans : (mkl::transpose)transpose_ct2, (int)transpose_ct2==0 ? mkl::transpose::trans : mkl::transpose::nontrans, N, N, alpha_S, d_A_S, N, d_B_S, N, beta_S, d_C_S, N).wait(), 0);
  //CHECK-NEXT:}
  a = cublasSsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans1, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  //CHECK:{
  //CHECK-NEXT:auto transpose_ct2 = trans1;
  //CHECK-NEXT:mkl::blas::gemmt(handle, (int)fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, (int)transpose_ct2==2 ? mkl::transpose::conjtrans : (mkl::transpose)transpose_ct2, (int)transpose_ct2==0 ? mkl::transpose::trans : mkl::transpose::nontrans, N, N, alpha_D, d_A_D, N, d_B_D, N, beta_D, d_C_D, N).wait();
  //CHECK-NEXT:}
  cublasDsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans1, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);

}
