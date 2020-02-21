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
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
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
  
  //CHECK:mkl::blas::rotm(dpct::get_default_queue_wait(), n, result_S, n, result_S, n, const_cast<float*>(x_S)).wait();
  cublasSrotm(n, result_S, n, result_S, n, x_S);
  //CHECK:mkl::blas::rotm(dpct::get_default_queue_wait(), n, result_D, n, result_D, n, const_cast<double*>(x_D)).wait();
  cublasDrotm(n, result_D, n, result_D, n, x_D);

  // CHECK:mkl::blas::copy(dpct::get_default_queue_wait(), n, x_S, incx, result_S, incy).wait();
  cublasScopy(n, x_S, incx, result_S, incy);
  // CHECK:mkl::blas::copy(dpct::get_default_queue_wait(), n, x_D, incx, result_D, incy).wait();
  cublasDcopy(n, x_D, incx, result_D, incy);
  // CHECK:mkl::blas::copy(dpct::get_default_queue_wait(), n, (std::complex<float>*)(x_C), incx, (std::complex<float>*)(result_C), incy).wait();
  cublasCcopy(n, x_C, incx, result_C, incy);
  // CHECK:mkl::blas::copy(dpct::get_default_queue_wait(), n, (std::complex<double>*)(x_Z), incx, (std::complex<double>*)(result_Z), incy).wait();
  cublasZcopy(n, x_Z, incx, result_Z, incy);

  // CHECK:mkl::blas::axpy(dpct::get_default_queue_wait(), n, alpha_S, x_S, incx, result_S, incy).wait();
  cublasSaxpy(n, alpha_S, x_S, incx, result_S, incy);
  // CHECK:mkl::blas::axpy(dpct::get_default_queue_wait(), n, alpha_D, x_D, incx, result_D, incy).wait();
  cublasDaxpy(n, alpha_D, x_D, incx, result_D, incy);
  // CHECK:mkl::blas::axpy(dpct::get_default_queue_wait(), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(x_C), incx, (std::complex<float>*)(result_C), incy).wait();
  cublasCaxpy(n, alpha_C, x_C, incx, result_C, incy);
  // CHECK:mkl::blas::axpy(dpct::get_default_queue_wait(), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(x_Z), incx, (std::complex<double>*)(result_Z), incy).wait();
  cublasZaxpy(n, alpha_Z, x_Z, incx, result_Z, incy);

  // CHECK:mkl::blas::scal(dpct::get_default_queue_wait(), n, alpha_S, result_S, incx).wait();
  cublasSscal(n, alpha_S, result_S, incx);
  // CHECK:mkl::blas::scal(dpct::get_default_queue_wait(), n, alpha_D, result_D, incx).wait();
  cublasDscal(n, alpha_D, result_D, incx);
  // CHECK:mkl::blas::scal(dpct::get_default_queue_wait(), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(result_C), incx).wait();
  cublasCscal(n, alpha_C, result_C, incx);
  // CHECK:mkl::blas::scal(dpct::get_default_queue_wait(), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(result_Z), incx).wait();
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

  // CHECK:mkl::blas::gemv(dpct::get_default_queue_wait(), mkl::transpose::nontrans, m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy).wait();
  cublasSgemv('N', m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);
  // CHECK:mkl::blas::gemv(dpct::get_default_queue_wait(), mkl::transpose::nontrans, m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy).wait();
  cublasDgemv('N', m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);
  // CHECK:mkl::blas::gemv(dpct::get_default_queue_wait(), mkl::transpose::nontrans, m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(x_C), lda, (std::complex<float>*)(y_C), incx, std::complex<float>((beta_C).x(),(beta_C).y()), (std::complex<float>*)(result_C), incy).wait();
  cublasCgemv('N', m, n, alpha_C, x_C, lda, y_C, incx, beta_C, result_C, incy);
  // CHECK:mkl::blas::gemv(dpct::get_default_queue_wait(), mkl::transpose::nontrans, m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(x_Z), lda, (std::complex<double>*)(y_Z), incx, std::complex<double>((beta_Z).x(),(beta_Z).y()), (std::complex<double>*)(result_Z), incy).wait();
  cublasZgemv('N', m, n, alpha_Z, x_Z, lda, y_Z, incx, beta_Z, result_Z, incy);

  // CHECK:mkl::blas::ger(dpct::get_default_queue_wait(), m, n, alpha_S, x_S, incx, y_S, incy, result_S, lda).wait();
  cublasSger(m, n, alpha_S, x_S, incx, y_S, incy, result_S, lda);
  // CHECK:mkl::blas::ger(dpct::get_default_queue_wait(), m, n, alpha_D, x_D, incx, y_D, incy, result_D, lda).wait();
  cublasDger(m, n, alpha_D, x_D, incx, y_D, incy, result_D, lda);
  // CHECK:mkl::blas::geru(dpct::get_default_queue_wait(), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(x_C), incx, (std::complex<float>*)(y_C), incy, (std::complex<float>*)(result_C), lda).wait();
  cublasCgeru(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK:mkl::blas::gerc(dpct::get_default_queue_wait(), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(x_C), incx, (std::complex<float>*)(y_C), incy, (std::complex<float>*)(result_C), lda).wait();
  cublasCgerc(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK:mkl::blas::geru(dpct::get_default_queue_wait(), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(x_Z), incx, (std::complex<double>*)(y_Z), incy, (std::complex<double>*)(result_Z), lda).wait();
  cublasZgeru(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);
  // CHECK:mkl::blas::gerc(dpct::get_default_queue_wait(), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(x_Z), incx, (std::complex<double>*)(y_Z), incy, (std::complex<double>*)(result_Z), lda).wait();
  cublasZgerc(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);

  //level 3

  //CHECK:mkl::blas::gemm(dpct::get_default_queue_wait(), mkl::transpose::nontrans, mkl::transpose::nontrans, n, n, n, alpha_S, A_S, n, B_S, n, beta_S, C_S, n).wait();
  cublasSgemm('N', 'N', n, n, n, alpha_S, A_S, n, B_S, n, beta_S, C_S, n);
  //CHECK:mkl::blas::gemm(dpct::get_default_queue_wait(), mkl::transpose::nontrans, mkl::transpose::nontrans, n, n, n, alpha_D, A_D, n, B_D, n, beta_D, C_D, n).wait();
  cublasDgemm('N', 'N', n, n, n, alpha_D, A_D, n, B_D, n, beta_D, C_D, n);
  //CHECK:mkl::blas::gemm(dpct::get_default_queue_wait(), mkl::transpose::nontrans, mkl::transpose::nontrans, n, n, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(A_C), n, (std::complex<float>*)(B_C), n, std::complex<float>((beta_C).x(),(beta_C).y()), (std::complex<float>*)(C_C), n).wait();
  cublasCgemm('N', 'N', n, n, n, alpha_C, A_C, n, B_C, n, beta_C, C_C, n);
  //CHECK:mkl::blas::gemm(dpct::get_default_queue_wait(), mkl::transpose::nontrans, mkl::transpose::nontrans, n, n, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(A_Z), n, (std::complex<double>*)(B_Z), n, std::complex<double>((beta_Z).x(),(beta_Z).y()), (std::complex<double>*)(C_Z), n).wait();
  cublasZgemm('N', 'N', n, n, n, alpha_Z, A_Z, n, B_Z, n, beta_Z, C_Z, n);

  //CHECK:mkl::blas::trmm(dpct::get_default_queue_wait(), mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, n, alpha_S, A_S, n, C_S, n).wait();
  cublasStrmm('L', 'L', 'N', 'N', n, n, alpha_S, A_S, n, C_S, n);
  //CHECK:mkl::blas::trmm(dpct::get_default_queue_wait(), mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, n, alpha_D, A_D, n, C_D, n).wait();
  cublasDtrmm('L', 'L', 'N', 'N', n, n, alpha_D, A_D, n, C_D, n);
  //CHECK:mkl::blas::trmm(dpct::get_default_queue_wait(), mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), (std::complex<float>*)(A_C), n,  (std::complex<float>*)(C_C), n).wait();
  cublasCtrmm('L', 'L', 'N', 'N', n, n, alpha_C, A_C, n, C_C, n);
  //CHECK:mkl::blas::trmm(dpct::get_default_queue_wait(), mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), (std::complex<double>*)(A_Z), n,  (std::complex<double>*)(C_Z), n).wait();
  cublasZtrmm('L', 'L', 'N', 'N', n, n, alpha_Z, A_Z, n, C_Z, n);
}
