// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-usm-legacy.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas.h>
#include <cuda_runtime.h>


int main() {
  int n = 275;
  int m = 275;
  int k = 275;
  int lda = 275;
  int ldb = 275;
  int ldc = 275;

  const float *A_S = 0;
  const float *B_S = 0;
  float *C_S = 0;
  float alpha_S = 1.0f;
  float beta_S = 0.0f;
  const double *A_D = 0;
  const double *B_D = 0;
  double *C_D = 0;
  double alpha_D = 1.0;
  double beta_D = 0.0;

  const float2 *A_C;
  const float2 *B_C;
  float2 *C_C;
  float2 alpha_C;
  float2 beta_C;
  const double2 *A_Z;
  const double2 *B_Z;
  double2 *C_Z;
  double2 alpha_Z;
  double2 beta_Z;

  const float *x_S = 0;
  const double *x_D = 0;
  const float *y_S = 0;
  const double *y_D = 0;
  const float2 *x_C;
  const float2 *y_C;
  const double2 *x_Z;
  const double2 *y_Z;

  int incx = 1;
  int incy = 1;
  int *result = 0;
  float *result_S = 0;
  double *result_D = 0;
  float2 *result_C;
  double2 *result_Z;

  int elemSize = 4;

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT: */
  // CHECK-NEXT: int status = (C_S = (float *)sycl::malloc_device((n)*(elemSize), dpct::get_current_device(), dpct::get_default_context()), 0);
  // CHECK-NEXT: C_S = (float *)sycl::malloc_device((n)*(elemSize), dpct::get_current_device(), dpct::get_default_context());
  cublasStatus status = cublasAlloc(n, elemSize, (void **)&C_S);
  cublasAlloc(n, elemSize, (void **)&C_S);

  // level 1

  // CHECK: int res;
  // CHECK-NEXT: {
  // CHECK-NEXT: int64_t result_temp_value;
  // CHECK-NEXT: int64_t* result_temp_ptr = (int64_t*) sycl::malloc_device(sizeof(int64_t), dpct::get_current_device(), dpct::get_default_context());
  // CHECK-NEXT: mkl::blas::iamax(dpct::get_default_queue_wait(), n, x_S, incx, result_temp_ptr).wait();
  // CHECK-NEXT: dpct::get_default_queue_wait().memcpy(&result_temp_value, result_temp_ptr, sizeof(int64_t)).wait();
  // CHECK-NEXT: res = result_temp_value;
  // CHECK-NEXT: }
  int res = cublasIsamax(n, x_S, incx);
  // CHECK: {
  // CHECK-NEXT: int64_t result_temp_value;
  // CHECK-NEXT: int64_t* result_temp_ptr = (int64_t*) sycl::malloc_device(sizeof(int64_t), dpct::get_current_device(), dpct::get_default_context());
  // CHECK-NEXT: mkl::blas::iamax(dpct::get_default_queue_wait(), n, x_D, incx, result_temp_ptr).wait();
  // CHECK-NEXT: dpct::get_default_queue_wait().memcpy(&result_temp_value, result_temp_ptr, sizeof(int64_t)).wait();
  // CHECK-NEXT: res = result_temp_value;
  // CHECK-NEXT: }
  res = cublasIdamax(n, x_D, incx);
  // CHECK: {
  // CHECK-NEXT: int64_t result_temp_value;
  // CHECK-NEXT: int64_t* result_temp_ptr = (int64_t*) sycl::malloc_device(sizeof(int64_t), dpct::get_current_device(), dpct::get_default_context());
  // CHECK-NEXT: mkl::blas::iamax(dpct::get_default_queue_wait(), n, (std::complex<float>*)(x_C), incx, result_temp_ptr).wait();
  // CHECK-NEXT: dpct::get_default_queue_wait().memcpy(&result_temp_value, result_temp_ptr, sizeof(int64_t)).wait();
  // CHECK-NEXT: res = result_temp_value;
  // CHECK-NEXT: }
  res = cublasIcamax(n, x_C, incx);
  // CHECK: {
  // CHECK-NEXT: int64_t result_temp_value;
  // CHECK-NEXT: int64_t* result_temp_ptr = (int64_t*) sycl::malloc_device(sizeof(int64_t), dpct::get_current_device(), dpct::get_default_context());
  // CHECK-NEXT: mkl::blas::iamax(dpct::get_default_queue_wait(), n, (std::complex<double>*)(x_Z), incx, result_temp_ptr).wait();
  // CHECK-NEXT: dpct::get_default_queue_wait().memcpy(&result_temp_value, result_temp_ptr, sizeof(int64_t)).wait();
  // CHECK-NEXT: res = result_temp_value;
  // CHECK-NEXT: }
  res = cublasIzamax(n, x_Z, incx);
  
  //CHECK:{
  //CHECK-NEXT:mkl::blas::rotm(dpct::get_default_queue_wait(), n, result_S, n, result_S, n, const_cast<float*>(x_S)).wait();
  //CHECK-NEXT:}
  cublasSrotm(n, result_S, n, result_S, n, x_S);
  //CHECK:{
  //CHECK-NEXT:mkl::blas::rotm(dpct::get_default_queue_wait(), n, result_D, n, result_D, n, const_cast<double*>(x_D)).wait();
  //CHECK-NEXT:}
  cublasDrotm(n, result_D, n, result_D, n, x_D);

  // CHECK: {
  // CHECK-NEXT: mkl::blas::copy(dpct::get_default_queue_wait(), n, x_S, incx, result_S, incy).wait();
  // CHECK-NEXT: }
  cublasScopy(n, x_S, incx, result_S, incy);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::copy(dpct::get_default_queue_wait(), n, x_D, incx, result_D, incy).wait();
  // CHECK-NEXT: }
  cublasDcopy(n, x_D, incx, result_D, incy);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::copy(dpct::get_default_queue_wait(), n, (std::complex<float>*)(x_C), incx, (std::complex<float>*)(result_C), incy).wait();
  // CHECK-NEXT: }
  cublasCcopy(n, x_C, incx, result_C, incy);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::copy(dpct::get_default_queue_wait(), n, (std::complex<double>*)(x_Z), incx, (std::complex<double>*)(result_Z), incy).wait();
  // CHECK-NEXT: }
  cublasZcopy(n, x_Z, incx, result_Z, incy);

  // CHECK: {
  // CHECK-NEXT: mkl::blas::axpy(dpct::get_default_queue_wait(), n, alpha_S, x_S, incx, result_S, incy).wait();
  // CHECK-NEXT: }
  cublasSaxpy(n, alpha_S, x_S, incx, result_S, incy);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::axpy(dpct::get_default_queue_wait(), n, alpha_D, x_D, incx, result_D, incy).wait();
  // CHECK-NEXT: }
  cublasDaxpy(n, alpha_D, x_D, incx, result_D, incy);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::axpy(dpct::get_default_queue_wait(), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(x_C), incx, (std::complex<float>*)(result_C), incy).wait();
  // CHECK-NEXT: }
  cublasCaxpy(n, alpha_C, x_C, incx, result_C, incy);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::axpy(dpct::get_default_queue_wait(), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(x_Z), incx, (std::complex<double>*)(result_Z), incy).wait();
  // CHECK-NEXT: }
  cublasZaxpy(n, alpha_Z, x_Z, incx, result_Z, incy);

  // CHECK: {
  // CHECK-NEXT: mkl::blas::scal(dpct::get_default_queue_wait(), n, alpha_S, result_S, incx).wait();
  // CHECK-NEXT: }
  cublasSscal(n, alpha_S, result_S, incx);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::scal(dpct::get_default_queue_wait(), n, alpha_D, result_D, incx).wait();
  // CHECK-NEXT: }
  cublasDscal(n, alpha_D, result_D, incx);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::scal(dpct::get_default_queue_wait(), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(result_C), incx).wait();
  // CHECK-NEXT: }
  cublasCscal(n, alpha_C, result_C, incx);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::scal(dpct::get_default_queue_wait(), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(result_Z), incx).wait();
  // CHECK-NEXT: }
  cublasZscal(n, alpha_Z, result_Z, incx);

  // CHECK: {
  // CHECK-NEXT: float result_temp_value;
  // CHECK-NEXT: float* result_temp_ptr = (float*) sycl::malloc_device(sizeof(float), dpct::get_current_device(), dpct::get_default_context());
  // CHECK-NEXT: mkl::blas::nrm2(dpct::get_default_queue_wait(), n, x_S, incx, result_temp_ptr).wait();
  // CHECK-NEXT: dpct::get_default_queue_wait().memcpy(&result_temp_value, result_temp_ptr, sizeof(float)).wait();
  // CHECK-NEXT: *result_S = result_temp_value;
  // CHECK-NEXT: }
  *result_S = cublasSnrm2(n, x_S, incx);
  // CHECK: {
  // CHECK-NEXT: double result_temp_value;
  // CHECK-NEXT: double* result_temp_ptr = (double*) sycl::malloc_device(sizeof(double), dpct::get_current_device(), dpct::get_default_context());
  // CHECK-NEXT: mkl::blas::nrm2(dpct::get_default_queue_wait(), n, x_D, incx, result_temp_ptr).wait();
  // CHECK-NEXT: dpct::get_default_queue_wait().memcpy(&result_temp_value, result_temp_ptr, sizeof(double)).wait();
  // CHECK-NEXT: *result_D = result_temp_value;
  // CHECK-NEXT: }
  *result_D = cublasDnrm2(n, x_D, incx);
  // CHECK: {
  // CHECK-NEXT: float result_temp_value;
  // CHECK-NEXT: float* result_temp_ptr = (float*) sycl::malloc_device(sizeof(float), dpct::get_current_device(), dpct::get_default_context());
  // CHECK-NEXT: mkl::blas::nrm2(dpct::get_default_queue_wait(), n, (std::complex<float>*)(x_C), incx, result_temp_ptr).wait();
  // CHECK-NEXT: dpct::get_default_queue_wait().memcpy(&result_temp_value, result_temp_ptr, sizeof(float)).wait();
  // CHECK-NEXT: *result_S = result_temp_value;
  // CHECK-NEXT: }
  *result_S = cublasScnrm2(n, x_C, incx);
  // CHECK: {
  // CHECK-NEXT: double result_temp_value;
  // CHECK-NEXT: double* result_temp_ptr = (double*) sycl::malloc_device(sizeof(double), dpct::get_current_device(), dpct::get_default_context());
  // CHECK-NEXT: mkl::blas::nrm2(dpct::get_default_queue_wait(), n, (std::complex<double>*)(x_Z), incx, result_temp_ptr).wait();
  // CHECK-NEXT: dpct::get_default_queue_wait().memcpy(&result_temp_value, result_temp_ptr, sizeof(double)).wait();
  // CHECK-NEXT: *result_D = result_temp_value;
  // CHECK-NEXT: }
  *result_D = cublasDznrm2(n, x_Z, incx);

  //level 2

  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: mkl::blas::gemv(dpct::get_default_queue_wait(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy).wait();
  // CHECK-NEXT: }
  cublasSgemv('N', m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);
  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: mkl::blas::gemv(dpct::get_default_queue_wait(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy).wait();
  // CHECK-NEXT: }
  cublasDgemv('N', m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);
  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: mkl::blas::gemv(dpct::get_default_queue_wait(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(x_C), lda, (std::complex<float>*)(y_C), incx, std::complex<float>((beta_C).x(),(beta_C).y()), (std::complex<float>*)(result_C), incy).wait();
  // CHECK-NEXT: }
  cublasCgemv('N', m, n, alpha_C, x_C, lda, y_C, incx, beta_C, result_C, incy);
  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: mkl::blas::gemv(dpct::get_default_queue_wait(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(x_Z), lda, (std::complex<double>*)(y_Z), incx, std::complex<double>((beta_Z).x(),(beta_Z).y()), (std::complex<double>*)(result_Z), incy).wait();
  // CHECK-NEXT: }
  cublasZgemv('N', m, n, alpha_Z, x_Z, lda, y_Z, incx, beta_Z, result_Z, incy);

  // CHECK: {
  // CHECK-NEXT: mkl::blas::ger(dpct::get_default_queue_wait(), m, n, alpha_S, x_S, incx, y_S, incy, result_S, lda).wait();
  // CHECK-NEXT: }
  cublasSger(m, n, alpha_S, x_S, incx, y_S, incy, result_S, lda);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::ger(dpct::get_default_queue_wait(), m, n, alpha_D, x_D, incx, y_D, incy, result_D, lda).wait();
  // CHECK-NEXT: }
  cublasDger(m, n, alpha_D, x_D, incx, y_D, incy, result_D, lda);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::geru(dpct::get_default_queue_wait(), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(x_C), incx, (std::complex<float>*)(y_C), incy, (std::complex<float>*)(result_C), lda).wait();
  // CHECK-NEXT: }
  cublasCgeru(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::gerc(dpct::get_default_queue_wait(), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(x_C), incx, (std::complex<float>*)(y_C), incy, (std::complex<float>*)(result_C), lda).wait();
  // CHECK-NEXT: }
  cublasCgerc(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::geru(dpct::get_default_queue_wait(), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(x_Z), incx, (std::complex<double>*)(y_Z), incy, (std::complex<double>*)(result_Z), lda).wait();
  // CHECK-NEXT: }
  cublasZgeru(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);
  // CHECK: {
  // CHECK-NEXT: mkl::blas::gerc(dpct::get_default_queue_wait(), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(x_Z), incx, (std::complex<double>*)(y_Z), incy, (std::complex<double>*)(result_Z), lda).wait();
  // CHECK-NEXT: }
  cublasZgerc(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);

  //level 3

  //CHECK:{
  //CHECK-NEXT:auto transpose_ct0 = 'N';
  //CHECK-NEXT:auto transpose_ct1 = 'N';
  //CHECK-NEXT:mkl::blas::gemm(dpct::get_default_queue_wait(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), n, n, n, alpha_S, A_S, n, B_S, n, beta_S, C_S, n).wait();
  //CHECK-NEXT:}
  cublasSgemm('N', 'N', n, n, n, alpha_S, A_S, n, B_S, n, beta_S, C_S, n);
  //CHECK:{
  //CHECK-NEXT:auto transpose_ct0 = 'N';
  //CHECK-NEXT:auto transpose_ct1 = 'N';
  //CHECK-NEXT:mkl::blas::gemm(dpct::get_default_queue_wait(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), n, n, n, alpha_D, A_D, n, B_D, n, beta_D, C_D, n).wait();
  //CHECK-NEXT:}
  cublasDgemm('N', 'N', n, n, n, alpha_D, A_D, n, B_D, n, beta_D, C_D, n);
  //CHECK:{
  //CHECK-NEXT:auto transpose_ct0 = 'N';
  //CHECK-NEXT:auto transpose_ct1 = 'N';
  //CHECK-NEXT:mkl::blas::gemm(dpct::get_default_queue_wait(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), n, n, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(A_C), n, (std::complex<float>*)(B_C), n, std::complex<float>((beta_C).x(),(beta_C).y()), (std::complex<float>*)(C_C), n).wait();
  //CHECK-NEXT:}
  cublasCgemm('N', 'N', n, n, n, alpha_C, A_C, n, B_C, n, beta_C, C_C, n);
  //CHECK:{
  //CHECK-NEXT:auto transpose_ct0 = 'N';
  //CHECK-NEXT:auto transpose_ct1 = 'N';
  //CHECK-NEXT:mkl::blas::gemm(dpct::get_default_queue_wait(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), n, n, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(A_Z), n, (std::complex<double>*)(B_Z), n, std::complex<double>((beta_Z).x(),(beta_Z).y()), (std::complex<double>*)(C_Z), n).wait();
  //CHECK-NEXT:}
  cublasZgemm('N', 'N', n, n, n, alpha_Z, A_Z, n, B_Z, n, beta_Z, C_Z, n);

  //CHECK:{
  //CHECK-NEXT:auto sidemode_ct0 = 'L';
  //CHECK-NEXT:auto fillmode_ct1 = 'L';
  //CHECK-NEXT:auto transpose_ct2 = 'N';
  //CHECK-NEXT:auto diagtype_ct3 = 'N';
  //CHECK-NEXT:mkl::blas::trmm(dpct::get_default_queue_wait(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct2)=='N'||(transpose_ct2)=='n')?(mkl::transpose::nontrans):(((transpose_ct2)=='T'||(transpose_ct2)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), (((diagtype_ct3)=='N'||(diagtype_ct3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, alpha_S, A_S, n,  C_S, n).wait();
  //CHECK-NEXT:}
  cublasStrmm('L', 'L', 'N', 'N', n, n, alpha_S, A_S, n, C_S, n);
  //CHECK:{
  //CHECK-NEXT:auto sidemode_ct0 = 'L';
  //CHECK-NEXT:auto fillmode_ct1 = 'L';
  //CHECK-NEXT:auto transpose_ct2 = 'N';
  //CHECK-NEXT:auto diagtype_ct3 = 'N';
  //CHECK-NEXT:mkl::blas::trmm(dpct::get_default_queue_wait(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct2)=='N'||(transpose_ct2)=='n')?(mkl::transpose::nontrans):(((transpose_ct2)=='T'||(transpose_ct2)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), (((diagtype_ct3)=='N'||(diagtype_ct3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, alpha_D, A_D, n,  C_D, n).wait();
  //CHECK-NEXT:}
  cublasDtrmm('L', 'L', 'N', 'N', n, n, alpha_D, A_D, n, C_D, n);
  //CHECK:{
  //CHECK-NEXT:auto sidemode_ct0 = 'L';
  //CHECK-NEXT:auto fillmode_ct1 = 'L';
  //CHECK-NEXT:auto transpose_ct2 = 'N';
  //CHECK-NEXT:auto diagtype_ct3 = 'N';
  //CHECK-NEXT:mkl::blas::trmm(dpct::get_default_queue_wait(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct2)=='N'||(transpose_ct2)=='n')?(mkl::transpose::nontrans):(((transpose_ct2)=='T'||(transpose_ct2)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), (((diagtype_ct3)=='N'||(diagtype_ct3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(A_C), n,  (std::complex<float>*)(C_C), n).wait();
  //CHECK-NEXT:}
  cublasCtrmm('L', 'L', 'N', 'N', n, n, alpha_C, A_C, n, C_C, n);
  //CHECK:{
  //CHECK-NEXT:auto sidemode_ct0 = 'L';
  //CHECK-NEXT:auto fillmode_ct1 = 'L';
  //CHECK-NEXT:auto transpose_ct2 = 'N';
  //CHECK-NEXT:auto diagtype_ct3 = 'N';
  //CHECK-NEXT:mkl::blas::trmm(dpct::get_default_queue_wait(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct2)=='N'||(transpose_ct2)=='n')?(mkl::transpose::nontrans):(((transpose_ct2)=='T'||(transpose_ct2)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), (((diagtype_ct3)=='N'||(diagtype_ct3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(A_Z), n,  (std::complex<double>*)(C_Z), n).wait();
  //CHECK-NEXT:}
  cublasZtrmm('L', 'L', 'N', 'N', n, n, alpha_Z, A_Z, n, C_Z, n);
}
