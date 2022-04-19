// RUN: c2s --format-range=none --usm-level=none -out-root %T/cublasLegacyCZ %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasLegacyCZ/cublasLegacyCZ.dp.cpp --match-full-lines %s

#include <cstdio>
#include <cublas.h>
#include <cuda_runtime.h>

char foo();

int main() {

  cublasStatus status;
  int n = 275;
  int m = 275;
  int k = 275;
  int lda = 275;
  int ldb = 275;
  int ldc = 275;
  const cuComplex *A_C = 0;
  const cuComplex *B_C = 0;
  cuComplex *C_C = 0;
  cuComplex alpha_C = make_cuComplex(1,0);
  cuComplex beta_C = make_cuComplex(0,0);
  const cuDoubleComplex *A_Z = 0;
  const cuDoubleComplex *B_Z = 0;
  cuDoubleComplex *C_Z = 0;
  cuDoubleComplex alpha_Z = make_cuDoubleComplex(1,0);
  cuDoubleComplex beta_Z = make_cuDoubleComplex(0,0);

  cuComplex *x_C = 0;
  cuDoubleComplex *x_Z = 0;
  cuComplex *y_C = 0;
  cuDoubleComplex *y_Z = 0;
  int incx = 1;
  int incy = 1;
  int *result = 0;
  cuComplex *result_C = 0;
  cuDoubleComplex *result_Z = 0;
  float *result_S = 0;
  double *result_D = 0;

  float *x_f = 0;
  float *y_f = 0;
  double *x_d = 0;
  double *y_d = 0;
  float *x_S = 0;
  float *y_S = 0;
  double *x_D = 0;
  double *y_D = 0;

  float alpha_S = 0;
  double alpha_D = 0;
  float beta_S = 0;
  double beta_D = 0;
  //level1

  //cublasI<t>amax
  // CHECK: int res;
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(*c2s::get_current_device().get_saved_queue(), n, x_C_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: res = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  int res = cublasIcamax(n, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(*c2s::get_current_device().get_saved_queue(), n, x_Z_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIzamax(n, x_Z, incx);

  //cublasI<t>amin
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(*c2s::get_current_device().get_saved_queue(), n, x_C_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIcamin(n, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(*c2s::get_current_device().get_saved_queue(), n, x_Z_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIzamin(n, x_Z, incx);

  //cublas<t>asum
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: sycl::buffer<float> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(*c2s::get_current_device().get_saved_queue(), n, x_C_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result_S = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasScasum(n, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: sycl::buffer<double> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(*c2s::get_current_device().get_saved_queue(), n, x_Z_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result_D = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDzasum(n, x_Z, incx);

  //cublas<t>dot
  // CHECK: sycl::float2 resCuComplex;
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(y_C);
  // CHECK-NEXT: sycl::buffer<std::complex<float>> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotu(*c2s::get_current_device().get_saved_queue(), n, x_C_buf_ct{{[0-9]+}}, incx, y_C_buf_ct{{[0-9]+}}, incy, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: resCuComplex = sycl::float2(res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0].real(), res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0].imag());
  // CHECK-NEXT: }
  cuComplex resCuComplex = cublasCdotu(n, x_C, incx, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(y_C);
  // CHECK-NEXT: sycl::buffer<std::complex<float>> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotc(*c2s::get_current_device().get_saved_queue(), n, x_C_buf_ct{{[0-9]+}}, incx, y_C_buf_ct{{[0-9]+}}, incy, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result_C = sycl::float2(res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0].real(), res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0].imag());
  // CHECK-NEXT: }
  *result_C = cublasCdotc(n, x_C, incx, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto y_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(y_Z);
  // CHECK-NEXT: sycl::buffer<std::complex<double>> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotu(*c2s::get_current_device().get_saved_queue(), n, x_Z_buf_ct{{[0-9]+}}, incx, y_Z_buf_ct{{[0-9]+}}, incy, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result_Z = sycl::double2(res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0].real(), res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0].imag());
  // CHECK-NEXT: }
  *result_Z = cublasZdotu(n, x_Z, incx, y_Z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto y_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(y_Z);
  // CHECK-NEXT: sycl::buffer<std::complex<double>> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotc(*c2s::get_current_device().get_saved_queue(), n, x_Z_buf_ct{{[0-9]+}}, incx, y_Z_buf_ct{{[0-9]+}}, incy, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result_Z = sycl::double2(res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0].real(), res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0].imag());
  // CHECK-NEXT: }
  *result_Z = cublasZdotc(n, x_Z, incx, y_Z, incy);

  //cublas<t>nrm2
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: sycl::buffer<float> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(*c2s::get_current_device().get_saved_queue(), n, x_C_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result_S = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasScnrm2(n, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: sycl::buffer<double> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(*c2s::get_current_device().get_saved_queue(), n, x_Z_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: *result_D = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDznrm2(n, x_Z, incx);




  //cublas<t>axpy
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto result_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(result_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::axpy(*c2s::get_current_device().get_saved_queue(), n, std::complex<float>(alpha_C.x(),alpha_C.y()), x_C_buf_ct{{[0-9]+}}, incx, result_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCaxpy(n, alpha_C, x_C, incx, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto result_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(result_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::axpy(*c2s::get_current_device().get_saved_queue(), n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), x_Z_buf_ct{{[0-9]+}}, incx, result_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZaxpy(n, alpha_Z, x_Z, incx, result_Z, incy);

  //cublas<t>copy
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto result_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(result_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::copy(*c2s::get_current_device().get_saved_queue(), n, x_C_buf_ct{{[0-9]+}}, incx, result_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCcopy(n, x_C, incx, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto result_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(result_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::copy(*c2s::get_current_device().get_saved_queue(), n, x_Z_buf_ct{{[0-9]+}}, incx, result_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZcopy(n, x_Z, incx, result_Z, incy);


  //cublas<t>rot
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(y_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rot(*c2s::get_current_device().get_saved_queue(), n, x_C_buf_ct{{[0-9]+}}, incx, y_C_buf_ct{{[0-9]+}}, incy, *x_S, *y_S);
  // CHECK-NEXT: }
  cublasCsrot(n, x_C, incx, y_C, incy, *x_S, *y_S);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto y_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(y_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rot(*c2s::get_current_device().get_saved_queue(), n, x_Z_buf_ct{{[0-9]+}}, incx, y_Z_buf_ct{{[0-9]+}}, incy, *x_D, *y_D);
  // CHECK-NEXT: }
  cublasZdrot(n, x_Z, incx, y_Z, incy, *x_D, *y_D);


  //cublas<t>scal
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(*c2s::get_current_device().get_saved_queue(), n, std::complex<float>(alpha_C.x(),alpha_C.y()), x_C_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  cublasCscal(n, alpha_C, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(*c2s::get_current_device().get_saved_queue(), n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), x_Z_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  cublasZscal(n, alpha_Z, x_Z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(*c2s::get_current_device().get_saved_queue(), n, alpha_S, x_C_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  cublasCsscal(n, alpha_S, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(*c2s::get_current_device().get_saved_queue(), n, alpha_D, x_Z_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  cublasZdscal(n, alpha_D, x_Z, incx);

  //cublas<t>swap
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(y_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::swap(*c2s::get_current_device().get_saved_queue(), n, x_C_buf_ct{{[0-9]+}}, incx, y_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCswap(n, x_C, incx, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto y_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(y_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::swap(*c2s::get_current_device().get_saved_queue(), n, x_Z_buf_ct{{[0-9]+}}, incx, y_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZswap(n, x_Z, incx, y_Z, incy);

  //level2
  //cublas<t>gbmv
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(y_C);
  // CHECK-NEXT: auto result_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(result_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gbmv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, m, n, m, n, std::complex<float>(alpha_C.x(),alpha_C.y()), x_C_buf_ct{{[0-9]+}}, lda, y_C_buf_ct{{[0-9]+}}, incx, std::complex<float>(beta_C.x(),beta_C.y()), result_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCgbmv('N', m, n, m, n, alpha_C, x_C, lda, y_C, incx, beta_C, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto y_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(y_Z);
  // CHECK-NEXT: auto result_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(result_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gbmv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, m, n, m, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), x_Z_buf_ct{{[0-9]+}}, lda, y_Z_buf_ct{{[0-9]+}}, incx, std::complex<double>(beta_Z.x(),beta_Z.y()), result_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZgbmv( 'N', m, n, m, n, alpha_Z, x_Z, lda, y_Z, incx, beta_Z, result_Z, incy);

  //cublas<t>gemv
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(y_C);
  // CHECK-NEXT: auto result_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(result_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, m, n, std::complex<float>(alpha_C.x(),alpha_C.y()), x_C_buf_ct{{[0-9]+}}, lda, y_C_buf_ct{{[0-9]+}}, incx, std::complex<float>(beta_C.x(),beta_C.y()), result_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCgemv('N', m, n, alpha_C, x_C, lda, y_C, incx, beta_C, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto y_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(y_Z);
  // CHECK-NEXT: auto result_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(result_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, m, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), x_Z_buf_ct{{[0-9]+}}, lda, y_Z_buf_ct{{[0-9]+}}, incx, std::complex<double>(beta_Z.x(),beta_Z.y()), result_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZgemv('N', m, n, alpha_Z, x_Z, lda, y_Z, incx, beta_Z, result_Z, incy);

  //cublas<t>ger
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(y_C);
  // CHECK-NEXT: auto result_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(result_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::geru(*c2s::get_current_device().get_saved_queue(), m, n, std::complex<float>(alpha_C.x(),alpha_C.y()), x_C_buf_ct{{[0-9]+}}, incx, y_C_buf_ct{{[0-9]+}}, incy, result_C_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasCgeru(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(y_C);
  // CHECK-NEXT: auto result_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(result_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gerc(*c2s::get_current_device().get_saved_queue(), m, n, std::complex<float>(alpha_C.x(),alpha_C.y()), x_C_buf_ct{{[0-9]+}}, incx, y_C_buf_ct{{[0-9]+}}, incy, result_C_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasCgerc(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto y_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(y_Z);
  // CHECK-NEXT: auto result_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(result_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::geru(*c2s::get_current_device().get_saved_queue(), m, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), x_Z_buf_ct{{[0-9]+}}, incx, y_Z_buf_ct{{[0-9]+}}, incy, result_Z_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasZgeru(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto y_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(y_Z);
  // CHECK-NEXT: auto result_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(result_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gerc(*c2s::get_current_device().get_saved_queue(), m, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), x_Z_buf_ct{{[0-9]+}}, incx, y_Z_buf_ct{{[0-9]+}}, incy, result_Z_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasZgerc(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);



  //cublas<t>tbmv
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto result_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(result_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbmv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, n, x_C_buf_ct{{[0-9]+}}, lda, result_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCtbmv('U', 'N', 'U', n, n, x_C, lda, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto result_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(result_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbmv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, n, x_Z_buf_ct{{[0-9]+}}, lda, result_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZtbmv('u', 'N', 'u', n, n, x_Z, lda, result_Z, incy);

  //cublas<t>tbsv
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto result_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(result_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbsv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, n, x_C_buf_ct{{[0-9]+}}, lda, result_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCtbsv('L', 'N', 'U', n, n, x_C, lda, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto result_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(result_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbsv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, n, x_Z_buf_ct{{[0-9]+}}, lda, result_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZtbsv('l', 'N', 'U', n, n, x_Z, lda, result_Z, incy);

  //cublas<t>tpmv
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto result_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(result_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpmv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_C_buf_ct{{[0-9]+}}, result_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCtpmv('U', 'N', 'U', n, x_C, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto result_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(result_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpmv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_Z_buf_ct{{[0-9]+}}, result_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZtpmv('U', 'N', 'U', n, x_Z, result_Z, incy);

  //cublas<t>tpsv
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto result_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(result_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpsv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_C_buf_ct{{[0-9]+}}, result_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCtpsv('U', 'N', 'U', n, x_C, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto result_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(result_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpsv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_Z_buf_ct{{[0-9]+}}, result_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZtpsv('U', 'N', 'U', n, x_Z, result_Z, incy);

  //cublas<t>trmv
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto result_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(result_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trmv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_C_buf_ct{{[0-9]+}}, lda, result_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCtrmv('U', 'N', 'U', n, x_C, lda, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto result_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(result_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trmv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_Z_buf_ct{{[0-9]+}}, lda, result_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZtrmv('U', 'N', 'U', n, x_Z, lda, result_Z, incy);

  //cublas<t>trsv
  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto result_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(result_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_C_buf_ct{{[0-9]+}}, lda, result_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCtrsv('U', 'N', 'U', n, x_C, lda, result_C, incy);


  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto result_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(result_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_Z_buf_ct{{[0-9]+}}, lda, result_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZtrsv('U', 'N', 'U', n, x_Z, lda, result_Z, incy);

  //chemv
  // CHECK: {
  // CHECK-NEXT: auto A_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(y_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hemv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, std::complex<float>(alpha_C.x(),alpha_C.y()), A_C_buf_ct{{[0-9]+}}, lda, x_C_buf_ct{{[0-9]+}}, incx, std::complex<float>(beta_C.x(),beta_C.y()), y_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasChemv ('U', n, alpha_C, A_C, lda, x_C, incx, beta_C, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto y_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(y_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hemv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), A_Z_buf_ct{{[0-9]+}}, lda, x_Z_buf_ct{{[0-9]+}}, incx, std::complex<double>(beta_Z.x(),beta_Z.y()), y_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZhemv ('U', n, alpha_Z, A_Z, lda, x_Z, incx, beta_Z, y_Z, incy);

  // CHECK: {
  // CHECK-NEXT: auto A_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(y_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hbmv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, k, std::complex<float>(alpha_C.x(),alpha_C.y()), A_C_buf_ct{{[0-9]+}}, lda, x_C_buf_ct{{[0-9]+}}, incx, std::complex<float>(beta_C.x(),beta_C.y()), y_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasChbmv ('U', n, k, alpha_C, A_C, lda, x_C, incx, beta_C, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto y_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(y_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hbmv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, k, std::complex<double>(alpha_Z.x(),alpha_Z.y()), A_Z_buf_ct{{[0-9]+}}, lda, x_Z_buf_ct{{[0-9]+}}, incx, std::complex<double>(beta_Z.x(),beta_Z.y()), y_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZhbmv ('U', n, k, alpha_Z, A_Z, lda, x_Z, incx, beta_Z, y_Z, incy);

  // CHECK: {
  // CHECK-NEXT: auto A_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(y_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpmv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, std::complex<float>(alpha_C.x(),alpha_C.y()), A_C_buf_ct{{[0-9]+}}, x_C_buf_ct{{[0-9]+}}, incx, std::complex<float>(beta_C.x(),beta_C.y()), y_C_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasChpmv('U', n, alpha_C, A_C, x_C, incx, beta_C, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto y_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(y_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpmv(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), A_Z_buf_ct{{[0-9]+}}, x_Z_buf_ct{{[0-9]+}}, incx, std::complex<double>(beta_Z.x(),beta_Z.y()), y_Z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZhpmv('U', n, alpha_Z, A_Z, x_Z, incx, beta_Z, y_Z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, alpha_S, x_C_buf_ct{{[0-9]+}}, incx, C_C_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasCher ('U', n, alpha_S, x_C, incx, C_C, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, alpha_D, x_Z_buf_ct{{[0-9]+}}, incx, C_Z_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasZher ('U', n, alpha_D, x_Z, incx, C_Z, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(y_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her2(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, std::complex<float>(alpha_C.x(),alpha_C.y()), x_C_buf_ct{{[0-9]+}}, incx, y_C_buf_ct{{[0-9]+}}, incy, C_C_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasCher2 ('U', n, alpha_C, x_C, incx, y_C, incy, C_C, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto y_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(y_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her2(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), x_Z_buf_ct{{[0-9]+}}, incx, y_Z_buf_ct{{[0-9]+}}, incy, C_Z_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasZher2 ('U', n, alpha_Z, x_Z, incx, y_Z, incy, C_Z, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpr(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, alpha_S, x_C_buf_ct{{[0-9]+}}, incx, C_C_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasChpr ('U', n, alpha_S, x_C, incx, C_C);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpr(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, alpha_D, x_Z_buf_ct{{[0-9]+}}, incx, C_Z_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasZhpr ('U', n, alpha_D, x_Z, incx, C_Z);

  // CHECK: {
  // CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(x_C);
  // CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(y_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpr2(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, std::complex<float>(alpha_C.x(),alpha_C.y()), x_C_buf_ct{{[0-9]+}}, incx, y_C_buf_ct{{[0-9]+}}, incy, C_C_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasChpr2 ('U', n, alpha_C, x_C, incx, y_C, incy, C_C);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(x_Z);
  // CHECK-NEXT: auto y_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(y_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpr2(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), x_Z_buf_ct{{[0-9]+}}, incx, y_Z_buf_ct{{[0-9]+}}, incy, C_Z_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasZhpr2 ('U', n, alpha_Z, x_Z, incx, y_Z, incy, C_Z);


  //level3
  // CHECK: {
  // CHECK-NEXT: auto A_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto B_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(B_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, m, n, k, std::complex<float>(alpha_C.x(),alpha_C.y()), A_C_buf_ct{{[0-9]+}}, lda, B_C_buf_ct{{[0-9]+}}, ldb, std::complex<float>(beta_C.x(),beta_C.y()), C_C_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCgemm('N', 'N', m, n, k, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto B_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(B_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, m, n, k, std::complex<double>(alpha_Z.x(),alpha_Z.y()), A_Z_buf_ct{{[0-9]+}}, lda, B_Z_buf_ct{{[0-9]+}}, ldb, std::complex<double>(beta_Z.x(),beta_Z.y()), C_Z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZgemm('N', 'N', m, n, k, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // cublas<T>symm
  // CHECK: {
  // CHECK-NEXT: auto A_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto B_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(B_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symm(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::side::right, oneapi::mkl::uplo::lower, m, n, std::complex<float>(alpha_C.x(),alpha_C.y()), A_C_buf_ct{{[0-9]+}}, lda, B_C_buf_ct{{[0-9]+}}, ldb, std::complex<float>(beta_C.x(),beta_C.y()), C_C_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCsymm('R', 'L', m, n, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto B_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(B_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symm(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::side::right, oneapi::mkl::uplo::lower, m, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), A_Z_buf_ct{{[0-9]+}}, lda, B_Z_buf_ct{{[0-9]+}}, ldb, std::complex<double>(beta_Z.x(),beta_Z.y()), C_Z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZsymm('r', 'L', m, n, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syrk(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, n, k, std::complex<float>(alpha_C.x(),alpha_C.y()), A_C_buf_ct{{[0-9]+}}, lda, std::complex<float>(beta_C.x(),beta_C.y()), C_C_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCsyrk('U', 'T', n, k, alpha_C, A_C, lda, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syrk(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, n, k, std::complex<double>(alpha_Z.x(),alpha_Z.y()), A_Z_buf_ct{{[0-9]+}}, lda, std::complex<double>(beta_Z.x(),beta_Z.y()), C_Z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZsyrk('U', 't', n, k, alpha_Z, A_Z, lda, beta_Z, C_Z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::herk(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, n, k, alpha_S, A_C_buf_ct{{[0-9]+}}, lda, beta_S, C_C_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCherk('U', 't', n, k, alpha_S, A_C, lda, beta_S, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::herk(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, n, k, alpha_D, A_Z_buf_ct{{[0-9]+}}, lda, beta_D, C_Z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZherk('U', 't', n, k, alpha_D, A_Z, lda, beta_D, C_Z, ldc);

  // cublas<T>syr2k
  // CHECK: {
  // CHECK-NEXT: auto A_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto B_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(B_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2k(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, n, k, std::complex<float>(alpha_C.x(),alpha_C.y()), A_C_buf_ct{{[0-9]+}}, lda, B_C_buf_ct{{[0-9]+}}, ldb, std::complex<float>(beta_C.x(),beta_C.y()), C_C_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCsyr2k('U', 'C', n, k, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto B_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(B_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2k(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, n, k, std::complex<double>(alpha_Z.x(),alpha_Z.y()), A_Z_buf_ct{{[0-9]+}}, lda, B_Z_buf_ct{{[0-9]+}}, ldb, std::complex<double>(beta_Z.x(),beta_Z.y()), C_Z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZsyr2k('U', 'c', n, k, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto B_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(B_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her2k(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, n, k, std::complex<float>(alpha_C.x(),alpha_C.y()), A_C_buf_ct{{[0-9]+}}, lda, B_C_buf_ct{{[0-9]+}}, ldb, beta_S, C_C_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCher2k('U', 'c', n, k, alpha_C, A_C, lda, B_C, ldb, beta_S, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto B_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(B_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her2k(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, n, k, std::complex<double>(alpha_Z.x(),alpha_Z.y()), A_Z_buf_ct{{[0-9]+}}, lda, B_Z_buf_ct{{[0-9]+}}, ldb, beta_D, C_Z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZher2k('U', 'c', n, k, alpha_Z, A_Z, lda, B_Z, ldb, beta_D, C_Z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto B_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(B_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hemm(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::side::right, oneapi::mkl::uplo::upper, m, n, std::complex<float>(alpha_C.x(),alpha_C.y()), A_C_buf_ct{{[0-9]+}}, lda, B_C_buf_ct{{[0-9]+}}, ldb, std::complex<float>(beta_C.x(),beta_C.y()), C_C_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasChemm ('R', 'U', m, n, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto B_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(B_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hemm(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::side::right, oneapi::mkl::uplo::upper, m, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), A_Z_buf_ct{{[0-9]+}}, lda, B_Z_buf_ct{{[0-9]+}}, ldb, std::complex<double>(beta_Z.x(),beta_Z.y()), C_Z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZhemm ('R', 'U', m, n, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // cublas<T>trsm
  // CHECK: {
  // CHECK-NEXT: auto A_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsm(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, std::complex<float>(alpha_C.x(),alpha_C.y()), A_C_buf_ct{{[0-9]+}}, lda, C_C_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCtrsm('L', 'U', 'N', 'n', m, n, alpha_C, A_C, lda, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsm(*c2s::get_current_device().get_saved_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), A_Z_buf_ct{{[0-9]+}}, lda, C_Z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZtrsm('l', 'U', 'N', 'N', m, n, alpha_Z, A_Z, lda, C_Z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct{{[0-9]+}} = foo();
  // CHECK-NEXT: auto fillmode_ct{{[0-9]+}} = foo();
  // CHECK-NEXT: auto transpose_ct{{[0-9]+}} = foo();
  // CHECK-NEXT: auto diagtype_ct{{[0-9]+}} = foo();
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = c2s::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsm(*c2s::get_current_device().get_saved_queue(), (sidemode_ct{{[0-9]+}}=='L'||sidemode_ct{{[0-9]+}}=='l') ? oneapi::mkl::side::left : oneapi::mkl::side::right, (fillmode_ct{{[0-9]+}}=='L'||fillmode_ct{{[0-9]+}}=='l') ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, (transpose_ct{{[0-9]+}}=='N'||transpose_ct{{[0-9]+}}=='n') ? oneapi::mkl::transpose::nontrans: ((transpose_ct{{[0-9]+}}=='T'||transpose_ct{{[0-9]+}}=='t') ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::conjtrans), (diagtype_ct{{[0-9]+}}=='N'||diagtype_ct{{[0-9]+}}=='n') ? oneapi::mkl::diag::nonunit : oneapi::mkl::diag::unit, m, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), A_Z_buf_ct{{[0-9]+}}, lda, C_Z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZtrsm(foo(), foo(), foo(), foo(), m, n, alpha_Z, A_Z, lda, C_Z, ldc);
}

