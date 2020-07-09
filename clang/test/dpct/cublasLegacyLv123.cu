// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasLegacyLv123.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas.h>
#include <cuda_runtime.h>

char foo();

cublasStatus status;
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

const float *x_S = 0;
const double *x_D = 0;
const float *y_S = 0;
const double *y_D = 0;
int incx = 1;
int incy = 1;
int *result = 0;
float *result_S = 0;
double *result_D = 0;


float *x_f = 0;
float *y_f = 0;
double *x_d = 0;
double *y_d = 0;

int main() {

  //level1

  //cublasI<t>amax
  // CHECK: int res;
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamax(*dpct::get_current_device().get_saved_queue(), n, x_S_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: res = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  int res = cublasIsamax(n, x_S, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamax(*dpct::get_current_device().get_saved_queue(), n, x_D_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIdamax(n, x_D, incx);

  //cublasI<t>amin
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamin(*dpct::get_current_device().get_saved_queue(), n, x_S_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIsamin(n, x_S, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamin(*dpct::get_current_device().get_saved_queue(), n, x_D_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIdamin(n, x_D, incx);

  //cublas<t>asum
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: sycl::buffer<float> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::asum(*dpct::get_current_device().get_saved_queue(), n, x_S_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result_S = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasSasum(n, x_S, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: sycl::buffer<double> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::asum(*dpct::get_current_device().get_saved_queue(), n, x_D_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result_D = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDasum(n, x_D, incx);

  //cublas<t>dot
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: sycl::buffer<float> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::dot(*dpct::get_current_device().get_saved_queue(), n, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result_S = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasSdot(n, x_S, incx, y_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: sycl::buffer<double> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::dot(*dpct::get_current_device().get_saved_queue(), n, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result_D = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDdot(n, x_D, incx, y_D, incy);

  //cublas<t>nrm2
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: sycl::buffer<float> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::nrm2(*dpct::get_current_device().get_saved_queue(), n, x_S_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result_S = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasSnrm2(n, x_S, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: sycl::buffer<double> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::nrm2(*dpct::get_current_device().get_saved_queue(), n, x_D_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result_D = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDnrm2(n, x_D, incx);




  //cublas<t>axpy
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::axpy(*dpct::get_current_device().get_saved_queue(), n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSaxpy(n, alpha_S, x_S, incx, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::axpy(*dpct::get_current_device().get_saved_queue(), n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDaxpy(n, alpha_D, x_D, incx, result_D, incy);

  //cublas<t>copy
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::copy(*dpct::get_current_device().get_saved_queue(), n, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasScopy(n, x_S, incx, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::copy(*dpct::get_current_device().get_saved_queue(), n, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDcopy(n, x_D, incx, result_D, incy);


  //cublas<t>rot
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: mkl::blas::rot(*dpct::get_current_device().get_saved_queue(), n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy, *x_S, *y_S);
  // CHECK-NEXT: }
  cublasSrot(n, x_f, incx, y_f, incy, *x_S, *y_S);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: mkl::blas::rot(*dpct::get_current_device().get_saved_queue(), n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy, *x_D, *y_D);
  // CHECK-NEXT: }
  cublasDrot(n, x_d, incx, y_d, incy, *x_D, *y_D);

  //cublas<t>rotg
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: mkl::blas::rotg(*dpct::get_current_device().get_saved_queue(), x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasSrotg(x_f, y_f, x_f, y_f);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: mkl::blas::rotg(*dpct::get_current_device().get_saved_queue(), x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasDrotg(x_d, y_d, x_d, y_d);

  //cublas<t>rotm
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: mkl::blas::rotm(*dpct::get_current_device().get_saved_queue(), n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy, x_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasSrotm(n, x_f, incx, y_f, incy, x_S);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: mkl::blas::rotm(*dpct::get_current_device().get_saved_queue(), n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy, x_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasDrotm(n, x_d, incx, y_d, incy, x_D);

  //cublas<t>rotmg
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: mkl::blas::rotmg(*dpct::get_current_device().get_saved_queue(), x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, *(x_S), y_f_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasSrotmg(x_f, y_f, y_f, x_S, y_f);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: mkl::blas::rotmg(*dpct::get_current_device().get_saved_queue(), x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, *(x_D), y_d_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasDrotmg(x_d, y_d, y_d, x_D, y_d);

  //cublas<t>scal
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: mkl::blas::scal(*dpct::get_current_device().get_saved_queue(), n, alpha_S, x_f_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  cublasSscal(n, alpha_S, x_f, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: mkl::blas::scal(*dpct::get_current_device().get_saved_queue(), n, alpha_D, x_d_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  cublasDscal(n, alpha_D, x_d, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: mkl::blas::swap(*dpct::get_current_device().get_saved_queue(), n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSswap(n, x_f, incx, y_f, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: mkl::blas::swap(*dpct::get_current_device().get_saved_queue(), n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDswap(n, x_d, incx, y_d, incy);

  //level2
  //cublas<t>gbmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::gbmv(*dpct::get_current_device().get_saved_queue(), mkl::transpose::nontrans, m, n, m, n, alpha_S, x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSgbmv('N', m, n, m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::gbmv(*dpct::get_current_device().get_saved_queue(), mkl::transpose::nontrans, m, n, m, n, alpha_D, x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDgbmv( 'N', m, n, m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>gemv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::gemv(*dpct::get_current_device().get_saved_queue(), mkl::transpose::nontrans, m, n, alpha_S, x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSgemv('N', m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::gemv(*dpct::get_current_device().get_saved_queue(), mkl::transpose::nontrans, m, n, alpha_D, x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDgemv('N', m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>ger
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::ger(*dpct::get_current_device().get_saved_queue(), m, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasSger(m, n, alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::ger(*dpct::get_current_device().get_saved_queue(), m, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasDger(m, n, alpha_D, x_D, incx, y_D, incy, result_D, lda);

  //cublas<t>sbmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::sbmv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, m, n, alpha_S, x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSsbmv('U', m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::sbmv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, m, n, alpha_D, x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDsbmv('U', m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>spmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::spmv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSspmv('U', n, alpha_S, x_S, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::spmv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDspmv('U', n, alpha_D, x_D, y_D, incx, beta_D, result_D, incy);

  //cublas<t>spr
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::spr(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasSspr('U', n, alpha_S, x_S, incx, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::spr(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasDspr('U', n, alpha_D, x_D, incx, result_D);

  //cublas<t>spr2
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::spr2(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasSspr2('U', n, alpha_S, x_S, incx, y_S, incy, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::spr2(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasDspr2('U', n, alpha_D, x_D, incx, y_D, incy, result_D);

  //cublas<t>symv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::symv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSsymv('U', n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::symv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDsymv('U', n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>syr
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::syr(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasSsyr('U', n, alpha_S, x_S, incx, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::syr(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasDsyr('U', n, alpha_D, x_D, incx, result_D, lda);

  //cublas<t>syr2
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::syr2(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasSsyr2('U', n, alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::syr2(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasDsyr2('U', n, alpha_D, x_D, incx, y_D, incy, result_D, lda);

  //cublas<t>tbmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::tbmv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasStbmv('U', 'N', 'U', n, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::tbmv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDtbmv('u', 'N', 'u', n, n, x_D, lda, result_D, incy);

  //cublas<t>tbsv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::tbsv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasStbsv('L', 'N', 'U', n, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::tbsv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDtbsv('l', 'N', 'U', n, n, x_D, lda, result_D, incy);

  //cublas<t>tpmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::tpmv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_buf_ct{{[0-9]+}}, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasStpmv('U', 'N', 'U', n, x_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::tpmv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_buf_ct{{[0-9]+}}, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDtpmv('U', 'N', 'U', n, x_D, result_D, incy);

  //cublas<t>tpsv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::tpsv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_buf_ct{{[0-9]+}}, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasStpsv('U', 'N', 'U', n, x_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::tpsv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_buf_ct{{[0-9]+}}, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDtpsv('U', 'N', 'U', n, x_D, result_D, incy);

  //cublas<t>trmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::trmv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasStrmv('U', 'N', 'U', n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::trmv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDtrmv('U', 'N', 'U', n, x_D, lda, result_D, incy);

  //cublas<t>trsv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::trsv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasStrsv('U', 'N', 'U', n, x_S, lda, result_S, incy);


  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::trsv(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDtrsv('U', 'N', 'U', n, x_D, lda, result_D, incy);

  //level3

  // cublas<T>symm
  // CHECK: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(B_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: mkl::blas::symm(*dpct::get_current_device().get_saved_queue(), mkl::side::right, mkl::uplo::lower, m, n, alpha_S, A_S_buf_ct{{[0-9]+}}, lda, B_S_buf_ct{{[0-9]+}}, ldb, beta_S, C_S_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasSsymm('R', 'L', m, n, alpha_S, A_S, lda, B_S, ldb, beta_S, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto B_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: mkl::blas::symm(*dpct::get_current_device().get_saved_queue(), mkl::side::right, mkl::uplo::lower, m, n, alpha_D, A_D_buf_ct{{[0-9]+}}, lda, B_D_buf_ct{{[0-9]+}}, ldb, beta_D, C_D_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasDsymm('r', 'L', m, n, alpha_D, A_D, lda, B_D, ldb, beta_D, C_D, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: mkl::blas::syrk(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::trans, n, k, alpha_S, A_S_buf_ct{{[0-9]+}}, lda, beta_S, C_S_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasSsyrk('U', 'T', n, k, alpha_S, A_S, lda, beta_S, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: mkl::blas::syrk(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::trans, n, k, alpha_D, A_D_buf_ct{{[0-9]+}}, lda, beta_D, C_D_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasDsyrk('U', 't', n, k, alpha_D, A_D, lda, beta_D, C_D, ldc);

  // cublas<T>syr2k
  // CHECK: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(B_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: mkl::blas::syr2k(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::conjtrans, n, k, alpha_S, A_S_buf_ct{{[0-9]+}}, lda, B_S_buf_ct{{[0-9]+}}, ldb, beta_S, C_S_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasSsyr2k('U', 'C', n, k, alpha_S, A_S, lda, B_S, ldb, beta_S, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto B_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: mkl::blas::syr2k(*dpct::get_current_device().get_saved_queue(), mkl::uplo::upper, mkl::transpose::conjtrans, n, k, alpha_D, A_D_buf_ct{{[0-9]+}}, lda, B_D_buf_ct{{[0-9]+}}, ldb, beta_D, C_D_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasDsyr2k('U', 'c', n, k, alpha_D, A_D, lda, B_D, ldb, beta_D, C_D, ldc);

  // cublas<T>trsm
  // CHECK: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: mkl::blas::trsm(*dpct::get_current_device().get_saved_queue(), mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, alpha_S, A_S_buf_ct{{[0-9]+}}, lda, C_S_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasStrsm('L', 'U', 'N', 'n', m, n, alpha_S, A_S, lda, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: mkl::blas::trsm(*dpct::get_current_device().get_saved_queue(), mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, alpha_D, A_D_buf_ct{{[0-9]+}}, lda, C_D_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasDtrsm('l', 'U', 'N', 'N', m, n, alpha_D, A_D, lda, C_D, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(B_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: mkl::blas::gemm(*dpct::get_current_device().get_saved_queue(), mkl::transpose::trans, mkl::transpose::conjtrans, n, n, n, alpha_S, A_S_buf_ct{{[0-9]+}}, n, B_S_buf_ct{{[0-9]+}}, n, beta_S, C_S_buf_ct{{[0-9]+}}, n);
  // CHECK-NEXT: }
  cublasSgemm('T', 'C', n, n, n, alpha_S, A_S, n, B_S, n, beta_S, C_S, n);

  // CHECK: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto B_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: mkl::blas::gemm(*dpct::get_current_device().get_saved_queue(), mkl::transpose::nontrans, mkl::transpose::nontrans, n, n, n, alpha_D, A_D_buf_ct{{[0-9]+}}, n, B_D_buf_ct{{[0-9]+}}, n, beta_D, C_D_buf_ct{{[0-9]+}}, n);
  // CHECK-NEXT: }
  cublasDgemm('N', 'n', n, n, n, alpha_D, A_D, n, B_D, n, beta_D, C_D, n);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct{{[0-9]+}} = foo();
  // CHECK-NEXT: auto fillmode_ct{{[0-9]+}} = foo();
  // CHECK-NEXT: auto transpose_ct{{[0-9]+}} = foo();
  // CHECK-NEXT: auto diagtype_ct{{[0-9]+}} = foo();
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: mkl::blas::trsm(*dpct::get_current_device().get_saved_queue(), (sidemode_ct{{[0-9]+}}=='L'||sidemode_ct{{[0-9]+}}=='l') ? mkl::side::left : mkl::side::right, (fillmode_ct{{[0-9]+}}=='L'||fillmode_ct{{[0-9]+}}=='l') ? mkl::uplo::lower : mkl::uplo::upper, (transpose_ct{{[0-9]+}}=='N'||transpose_ct{{[0-9]+}}=='n') ? mkl::transpose::nontrans: ((transpose_ct{{[0-9]+}}=='T'||transpose_ct{{[0-9]+}}=='t') ? mkl::transpose::trans : mkl::transpose::conjtrans), (diagtype_ct{{[0-9]+}}=='N'||diagtype_ct{{[0-9]+}}=='n') ? mkl::diag::nonunit : mkl::diag::unit, m, n, alpha_D, A_D_buf_ct{{[0-9]+}}, lda, C_D_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasDtrsm(foo(), foo(), foo(), foo(), m, n, alpha_D, A_D, lda, C_D, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto B_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: mkl::blas::gemm(*dpct::get_current_device().get_saved_queue(), mkl::transpose::nontrans, mkl::transpose::nontrans, n, n, n, alpha_D, A_D_buf_ct{{[0-9]+}}, n, B_D_buf_ct{{[0-9]+}}, n, beta_D, C_D_buf_ct{{[0-9]+}}, n);
  // CHECK-NEXT: }
  // CHECK-NEXT: for(;;){}
  for(cublasDgemm('N', 'n', n, n, n, alpha_D, A_D, n, B_D, n, beta_D, C_D, n);;){}

  // Because the return value of origin API is the result value, not the status, so keep using lambda here.
  // CHECK: for(int i = [&](){
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamax(*dpct::get_current_device().get_saved_queue(), n, x_S_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: return res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }();;){}
  for(int i = cublasIsamax(n, x_S, incx);;){}

}


// Because the return value of origin API is the result value, not the status, so keep using lambda here.
//CHECK:int bar(){
//CHECK-NEXT:  return [&](){
//CHECK-NEXT:  auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
//CHECK-NEXT:  sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
//CHECK-NEXT:  mkl::blas::iamax(*dpct::get_current_device().get_saved_queue(), n, x_S_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
//CHECK-NEXT:  return res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
//CHECK-NEXT:  }();
//CHECK-NEXT:}
int bar(){
  return cublasIsamax(n, x_S, incx);
}