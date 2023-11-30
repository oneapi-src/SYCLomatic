// RUN: dpct --format-range=none --usm-level=none -out-root %T/cublasRegularCZ %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasRegularCZ/cublasRegularCZ.dp.cpp --match-full-lines %s

#include <cuda_runtime.h>
#include <cublas_v2.h>

int foo();

int main(){
  cublasStatus_t status;
  cublasHandle_t handle;

  int* result = 0;
  float* result_f = 0;
  double* result_d = 0;
  cuComplex* x_c = 0;
  cuDoubleComplex* x_z = 0;

  int incx = 1;
  int incy = 1;
  int n = 10;

  //level 1
  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = sycl::buffer<int>(sycl::range<1>(1));
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result)) {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = sycl::buffer<int>(result, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamax(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}}, oneapi::mkl::index_base::one));
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = sycl::buffer<int>(sycl::range<1>(1));
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result)) {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = sycl::buffer<int>(result, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}}, oneapi::mkl::index_base::one);
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIcamax(handle, n, x_c, incx, result);
  cublasIcamax(handle, n, x_c, incx, result);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = sycl::buffer<int>(sycl::range<1>(1));
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result)) {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = sycl::buffer<int>(result, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamax(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}}, oneapi::mkl::index_base::one));
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = sycl::buffer<int>(sycl::range<1>(1));
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result)) {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = sycl::buffer<int>(result, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}}, oneapi::mkl::index_base::one);
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIzamax(handle, n, x_z, incx, result);
  cublasIzamax(handle, n, x_z, incx, result);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = sycl::buffer<int>(sycl::range<1>(1));
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result)) {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = sycl::buffer<int>(result, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamin(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}}, oneapi::mkl::index_base::one));
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = sycl::buffer<int>(sycl::range<1>(1));
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result)) {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = sycl::buffer<int>(result, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}}, oneapi::mkl::index_base::one);
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIcamin(handle, n, x_c, incx, result);
  cublasIcamin(handle, n, x_c, incx, result);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = sycl::buffer<int>(sycl::range<1>(1));
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result)) {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = sycl::buffer<int>(result, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamin(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}}, oneapi::mkl::index_base::one));
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = sycl::buffer<int>(sycl::range<1>(1));
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result)) {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = sycl::buffer<int>(result, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}}, oneapi::mkl::index_base::one);
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIzamin(handle, n, x_z, incx, result);
  cublasIzamin(handle, n, x_z, incx, result);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_f)) {
  // CHECK-NEXT:   result_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_f);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(result_f, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::asum(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, result_f_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_f)) {
  // CHECK-NEXT:   result_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_f);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(result_f, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, result_f_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasScasum(handle, n, x_c, incx, result_f);
  cublasScasum(handle, n, x_c, incx, result_f);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_d)) {
  // CHECK-NEXT:   result_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_d);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(result_d, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::asum(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, result_d_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_d)) {
  // CHECK-NEXT:   result_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_d);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(result_d, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, result_d_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDzasum(handle, n, x_z, incx, result_d);
  cublasDzasum(handle, n, x_z, incx, result_d);

  cuComplex* alpha_c = 0;
  cuComplex* beta_c = 0;
  cuDoubleComplex* alpha_z = 0;
  cuDoubleComplex* beta_z = 0;
  float* alpha_f = 0;
  double* alpha_d = 0;
  cuComplex* y_c = 0;
  cuDoubleComplex* y_z = 0;

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(*handle, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::axpy(*handle, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCaxpy(handle, n, alpha_c, x_c, incx, y_c, incy);
  cublasCaxpy(handle, n, alpha_c, x_c, incx, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(*handle, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::axpy(*handle, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZaxpy(handle, n, alpha_z, x_z, incx, y_z, incy);
  cublasZaxpy(handle, n, alpha_z, x_z, incx, y_z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::copy(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::copy(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCcopy(handle, n, x_c, incx, y_c, incy);
  cublasCcopy(handle, n, x_c, incx, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::copy(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::copy(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZcopy(handle, n, x_z, incx, y_z, incy);
  cublasZcopy(handle, n, x_z, incx, y_z, incy);

  cuComplex* result_c = 0;
  cuDoubleComplex* result_z = 0;

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_c)) {
  // CHECK-NEXT:   result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>((std::complex<float>*)result_c, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dotu(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_c)) {
  // CHECK-NEXT:   result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>((std::complex<float>*)result_c, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotu(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasCdotu(handle, n, x_c, incx, y_c, incy, result_c);
  cublasCdotu(handle, n, x_c, incx, y_c, incy, result_c);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_c)) {
  // CHECK-NEXT:   result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>((std::complex<float>*)result_c, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dotc(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_c)) {
  // CHECK-NEXT:   result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>((std::complex<float>*)result_c, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotc(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasCdotc(handle, n, x_c, incx, y_c, incy, result_c);
  cublasCdotc(handle, n, x_c, incx, y_c, incy, result_c);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_z)) {
  // CHECK-NEXT:   result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>((std::complex<double>*)result_z, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dotu(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_z)) {
  // CHECK-NEXT:   result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>((std::complex<double>*)result_z, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotu(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasZdotu(handle, n, x_z, incx, y_z, incy, result_z);
  cublasZdotu(handle, n, x_z, incx, y_z, incy, result_z);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_z)) {
  // CHECK-NEXT:   result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>((std::complex<double>*)result_z, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dotc(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_z)) {
  // CHECK-NEXT:   result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>((std::complex<double>*)result_z, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotc(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasZdotc(handle, n, x_z, incx, y_z, incy, result_z);
  cublasZdotc(handle, n, x_z, incx, y_z, incy, result_z);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_f)) {
  // CHECK-NEXT:   result_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_f);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(result_f, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::nrm2(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, result_f_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_f)) {
  // CHECK-NEXT:   result_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_f);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(result_f, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, result_f_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasScnrm2(handle, n, x_c, incx, result_f);
  cublasScnrm2(handle, n, x_c, incx, result_f);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_d)) {
  // CHECK-NEXT:   result_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_d);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(result_d, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::nrm2(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, result_d_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_d)) {
  // CHECK-NEXT:   result_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_d);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(result_d, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, result_d_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDznrm2(handle, n, x_z, incx, result_d);
  cublasDznrm2(handle, n, x_z, incx, result_d);

  float* c_f = 0;
  float* s_f = 0;
  double* c_d = 0;
  double* s_d = 0;
  cuComplex* c_c = 0;
  cuComplex* s_c = 0;
  cuDoubleComplex* c_z = 0;
  cuDoubleComplex* s_z = 0;

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, dpct::get_value(c_f, *handle), dpct::get_value(s_f, *handle)));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rot(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, dpct::get_value(c_f, *handle), dpct::get_value(s_f, *handle));
  // CHECK-NEXT: }
  status = cublasCsrot(handle, n, x_c, incx, y_c, incy, c_f, s_f);
  cublasCsrot(handle, n, x_c, incx, y_c, incy, c_f, s_f);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, dpct::get_value(c_d, *handle), dpct::get_value(s_d, *handle)));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rot(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, dpct::get_value(c_d, *handle), dpct::get_value(s_d, *handle));
  // CHECK-NEXT: }
  status = cublasZdrot(handle, n, x_z, incx, y_z, incy, c_d, s_d);
  cublasZdrot(handle, n, x_z, incx, y_z, incy, c_d, s_d);

  // CHECK: status = DPCT_CHECK_ERROR(dpct::rot(*handle, n, x_c, dpct::library_data_t::complex_float, incx, y_c, dpct::library_data_t::complex_float, incy, c_f, s_c, dpct::library_data_t::complex_float));
  // CHECK-NEXT: dpct::rot(*handle, n, x_c, dpct::library_data_t::complex_float, incx, y_c, dpct::library_data_t::complex_float, incy, c_f, s_c, dpct::library_data_t::complex_float);
  status = cublasCrot(handle, n, x_c, incx, y_c, incy, c_f, s_c);
  cublasCrot(handle, n, x_c, incx, y_c, incy, c_f, s_c);

  // CHECK: status = DPCT_CHECK_ERROR(dpct::rot(*handle, n, x_z, dpct::library_data_t::complex_double, incx, y_z, dpct::library_data_t::complex_double, incy, c_d, s_z, dpct::library_data_t::complex_double));
  // CHECK-NEXT: dpct::rot(*handle, n, x_z, dpct::library_data_t::complex_double, incx, y_z, dpct::library_data_t::complex_double, incy, c_d, s_z, dpct::library_data_t::complex_double);
  status = cublasZrot(handle, n, x_z, incx, y_z, incy, c_d, s_z);
  cublasZrot(handle, n, x_z, incx, y_z, incy, c_d, s_z);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>(sycl::range<1>(1));
  // CHECK-NEXT: auto c_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto s_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(x_c)) {
  // CHECK-NEXT:   x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT:   y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT:   c_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(c_f);
  // CHECK-NEXT:   s_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(s_c);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   x_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>((std::complex<float>*)x_c, sycl::range<1>(1));
  // CHECK-NEXT:   y_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>((std::complex<float>*)y_c, sycl::range<1>(1));
  // CHECK-NEXT:   c_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(c_f, sycl::range<1>(1));
  // CHECK-NEXT:   s_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>((std::complex<float>*)s_c, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rotg(*handle, x_c_buf_ct{{[0-9]+}}, y_c_buf_ct{{[0-9]+}}, c_f_buf_ct{{[0-9]+}}, s_c_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>(sycl::range<1>(1));
  // CHECK-NEXT: auto c_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto s_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(x_c)) {
  // CHECK-NEXT:   x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT:   y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT:   c_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(c_f);
  // CHECK-NEXT:   s_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(s_c);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   x_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>((std::complex<float>*)x_c, sycl::range<1>(1));
  // CHECK-NEXT:   y_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>((std::complex<float>*)y_c, sycl::range<1>(1));
  // CHECK-NEXT:   c_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(c_f, sycl::range<1>(1));
  // CHECK-NEXT:   s_c_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<float>>((std::complex<float>*)s_c, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotg(*handle, x_c_buf_ct{{[0-9]+}}, y_c_buf_ct{{[0-9]+}}, c_f_buf_ct{{[0-9]+}}, s_c_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasCrotg(handle, x_c, y_c, c_f, s_c);
  cublasCrotg(handle, x_c, y_c, c_f, s_c);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>(sycl::range<1>(1));
  // CHECK-NEXT: auto c_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto s_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(x_z)) {
  // CHECK-NEXT:   x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT:   y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT:   c_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(c_d);
  // CHECK-NEXT:   s_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(s_z);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   x_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>((std::complex<double>*)x_z, sycl::range<1>(1));
  // CHECK-NEXT:   y_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>((std::complex<double>*)y_z, sycl::range<1>(1));
  // CHECK-NEXT:   c_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(c_d, sycl::range<1>(1));
  // CHECK-NEXT:   s_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>((std::complex<double>*)s_z, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rotg(*handle, x_z_buf_ct{{[0-9]+}}, y_z_buf_ct{{[0-9]+}}, c_d_buf_ct{{[0-9]+}}, s_z_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>(sycl::range<1>(1));
  // CHECK-NEXT: auto c_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto s_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(x_z)) {
  // CHECK-NEXT:   x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT:   y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT:   c_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(c_d);
  // CHECK-NEXT:   s_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(s_z);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   x_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>((std::complex<double>*)x_z, sycl::range<1>(1));
  // CHECK-NEXT:   y_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>((std::complex<double>*)y_z, sycl::range<1>(1));
  // CHECK-NEXT:   c_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(c_d, sycl::range<1>(1));
  // CHECK-NEXT:   s_z_buf_ct{{[0-9]+}} = sycl::buffer<std::complex<double>>((std::complex<double>*)s_z, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotg(*handle, x_z_buf_ct{{[0-9]+}}, y_z_buf_ct{{[0-9]+}}, c_d_buf_ct{{[0-9]+}}, s_z_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasZrotg(handle, x_z, y_z, c_d, s_z);
  cublasZrotg(handle, x_z, y_z, c_d, s_z);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCscal(handle, n, alpha_c, x_c, incx);
  cublasCscal(handle, n, alpha_c, x_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZscal(handle, n, alpha_z, x_z, incx);
  cublasZscal(handle, n, alpha_z, x_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha_f, *handle), x_c_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha_f, *handle), x_c_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCsscal(handle, n, alpha_f, x_c, incx);
  cublasCsscal(handle, n, alpha_f, x_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha_d, *handle), x_z_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha_d, *handle), x_z_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZdscal(handle, n, alpha_d, x_z, incx);
  cublasZdscal(handle, n, alpha_d, x_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::swap(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::swap(*handle, n, x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCswap(handle, n, x_c, incx, y_c, incy);
  cublasCswap(handle, n, x_c, incx, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::swap(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::swap(*handle, n, x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZswap(handle, n, x_z, incx, y_z, incy);
  cublasZswap(handle, n, x_z, incx, y_z, incy);

  //level 2
  int m=0;
  int kl=0;
  int ku=0;
  int lda = 10;
  int trans0 = 0;
  int trans1 = 1;
  int trans2 = 2;
  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gbmv(*handle, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, m, n, kl, ku, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, lda, x_c_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_c, *handle), y_c_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gbmv(*handle, oneapi::mkl::transpose::nontrans, m, n, kl, ku, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, lda, x_c_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_c, *handle), y_c_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCgbmv(handle, (cublasOperation_t)trans0, m, n, kl, ku, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasCgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gbmv(*handle, trans1==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans1, m, n, kl, ku, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, lda, x_z_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_z, *handle), y_z_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gbmv(*handle, oneapi::mkl::transpose::nontrans, m, n, kl, ku, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, lda, x_z_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_z, *handle), y_z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZgbmv(handle, (cublasOperation_t)trans1, m, n, kl, ku, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemv(*handle, trans2==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans2, m, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, lda, x_c_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_c, *handle), y_c_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemv(*handle, oneapi::mkl::transpose::nontrans, m, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, lda, x_c_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_c, *handle), y_c_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCgemv(handle, (cublasOperation_t)trans2, m, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasCgemv(handle, CUBLAS_OP_N, m, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemv(*handle, oneapi::mkl::transpose::nontrans, m, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, lda, x_z_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_z, *handle), y_z_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemv(*handle, oneapi::mkl::transpose::nontrans, m, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, lda, x_z_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_z, *handle), y_z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZgemv(handle, (cublasOperation_t)0, m, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZgemv(handle, CUBLAS_OP_N, m, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::geru(*handle, m, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::geru(*handle, m, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCgeru(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCgeru(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gerc(*handle, m, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gerc(*handle, m, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCgerc(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCgerc(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::geru(*handle, m, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::geru(*handle, m, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZgeru(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZgeru(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gerc(*handle, m, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gerc(*handle, m, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZgerc(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZgerc(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  int k = 1;
  int fill0 = 0;
  int fill1 = 1;
  int diag0 = 0;
  int diag1 = 1;
  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, (oneapi::mkl::diag)diag0, n, k, x_c_buf_ct{{[0-9]+}}, lda, result_c_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbmv(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, k, x_c_buf_ct{{[0-9]+}}, lda, result_c_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtbmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)1, (cublasDiagType_t)diag0, n, k, x_c, lda, result_c, incx);
  cublasCtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_c, lda, result_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbmv(*handle, fill1==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, (oneapi::mkl::diag)diag1, n, k, x_z_buf_ct{{[0-9]+}}, lda, result_z_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbmv(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, k, x_z_buf_ct{{[0-9]+}}, lda, result_z_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtbmv(handle, (cublasFillMode_t)fill1, (cublasOperation_t)2, (cublasDiagType_t)diag1, n, k, x_z, lda, result_z, incx);
  cublasZtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_z, lda, result_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbsv(*handle, oneapi::mkl::uplo::lower, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, oneapi::mkl::diag::nonunit,  n, k, x_c_buf_ct{{[0-9]+}}, lda, result_c_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbsv(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit,  n, k, x_c_buf_ct{{[0-9]+}}, lda, result_c_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtbsv(handle, (cublasFillMode_t)0, (cublasOperation_t)trans0, (cublasDiagType_t)0,  n, k, x_c, lda, result_c, incx);
  cublasCtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_c, lda, result_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbsv(*handle, oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, oneapi::mkl::diag::unit,  n, k, x_z_buf_ct{{[0-9]+}}, lda, result_z_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbsv(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit,  n, k, x_z_buf_ct{{[0-9]+}}, lda, result_z_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtbsv(handle, (cublasFillMode_t)1, (cublasOperation_t)trans0, (cublasDiagType_t)1,  n, k, x_z, lda, result_z, incx);
  cublasZtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_z, lda, result_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_c_buf_ct{{[0-9]+}}, result_c_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpmv(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, x_c_buf_ct{{[0-9]+}}, result_c_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, result_c, incx);
  cublasCtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_z_buf_ct{{[0-9]+}}, result_z_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpmv(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, x_z_buf_ct{{[0-9]+}}, result_z_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, result_z, incx);
  cublasZtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpsv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_c_buf_ct{{[0-9]+}}, result_c_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpsv(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, x_c_buf_ct{{[0-9]+}}, result_c_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, result_c, incx);
  cublasCtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpsv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_z_buf_ct{{[0-9]+}}, result_z_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpsv(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, x_z_buf_ct{{[0-9]+}}, result_z_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, result_z, incx);
  cublasZtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_c_buf_ct{{[0-9]+}}, lda, result_c_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trmv(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, x_c_buf_ct{{[0-9]+}}, lda, result_c_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, lda, result_c, incx);
  cublasCtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_z_buf_ct{{[0-9]+}}, lda, result_z_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trmv(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, x_z_buf_ct{{[0-9]+}}, lda, result_z_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, lda, result_z, incx);
  cublasZtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_c_buf_ct{{[0-9]+}}, lda, result_c_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsv(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, x_c_buf_ct{{[0-9]+}}, lda, result_c_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, lda, result_c, incx);
  cublasCtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_z_buf_ct{{[0-9]+}}, lda, result_z_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsv(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, x_z_buf_ct{{[0-9]+}}, lda, result_z_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, lda, result_z, incx);
  cublasZtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, lda, x_c_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_c, *handle), y_c_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hemv(*handle, oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, lda, x_c_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_c, *handle), y_c_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasChemv(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasChemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, lda, x_z_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_z, *handle), y_z_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hemv(*handle, oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, lda, x_z_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_z, *handle), y_z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZhemv(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZhemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hbmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, k, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, lda, x_c_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_c, *handle), y_c_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hbmv(*handle, oneapi::mkl::uplo::lower, n, k, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, lda, x_c_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_c, *handle), y_c_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasChbmv(handle, (cublasFillMode_t)fill0, n, k, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasChbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hbmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, k, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, lda, x_z_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_z, *handle), y_z_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hbmv(*handle, oneapi::mkl::uplo::lower, n, k, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, lda, x_z_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_z, *handle), y_z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZhbmv(handle, (cublasFillMode_t)fill0, n, k, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZhbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, x_c_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_c, *handle), y_c_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpmv(*handle, oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, x_c_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_c, *handle), y_c_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasChpmv(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, x_c, incx, beta_c, y_c, incy);
  cublasChpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, x_c, incx, beta_c, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, x_z_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_z, *handle), y_z_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpmv(*handle, oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, x_z_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_z, *handle), y_z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZhpmv(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, x_z, incx, beta_z, y_z, incy);
  cublasZhpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, x_z, incx, beta_z, y_z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_f, *handle), x_c_buf_ct{{[0-9]+}}, incx, result_c_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her(*handle, oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_f, *handle), x_c_buf_ct{{[0-9]+}}, incx, result_c_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCher(handle, (cublasFillMode_t)fill0, n, alpha_f, x_c, incx, result_c, lda);
  cublasCher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_d, *handle), x_z_buf_ct{{[0-9]+}}, incx, result_z_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her(*handle, oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_d, *handle), x_z_buf_ct{{[0-9]+}}, incx, result_z_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZher(handle, (cublasFillMode_t)fill0, n, alpha_d, x_z, incx, result_z, lda);
  cublasZher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her2(*handle, oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCher2(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her2(*handle, oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZher2(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpr(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_f, *handle), x_c_buf_ct{{[0-9]+}}, incx, result_c_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpr(*handle, oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_f, *handle), x_c_buf_ct{{[0-9]+}}, incx, result_c_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasChpr(handle, (cublasFillMode_t)fill0, n, alpha_f, x_c, incx, result_c);
  cublasChpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpr(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_d, *handle), x_z_buf_ct{{[0-9]+}}, incx, result_z_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpr(*handle, oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_d, *handle), x_z_buf_ct{{[0-9]+}}, incx, result_z_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasZhpr(handle, (cublasFillMode_t)fill0, n, alpha_d, x_z, incx, result_z);
  cublasZhpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpr2(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpr2(*handle, oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasChpr2(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, y_c, incy, result_c);
  cublasChpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpr2(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpr2(*handle, oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasZhpr2(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, y_z, incy, result_z);
  cublasZhpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, lda, y_c_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_c, *handle), result_c_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symv(*handle, oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, lda, y_c_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_c, *handle), result_c_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCsymv(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, lda, y_c, incx, beta_c, result_c, incy);
  cublasCsymv(handle, CUBLAS_FILL_MODE_UPPER, n, alpha_c, x_c, lda, y_c, incx, beta_c, result_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, lda, y_z_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_z, *handle), result_z_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symv(*handle, oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, lda, y_z_buf_ct{{[0-9]+}}, incx, dpct::get_value(beta_z, *handle), result_z_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZsymv(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, lda, y_z, incx, beta_z, result_z, incy);
  cublasZsymv(handle, CUBLAS_FILL_MODE_UPPER, n, alpha_z, x_z, lda, y_z, incx, beta_z, result_z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, result_c_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr(*handle, oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, result_c_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCsyr(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, result_c, lda);
  cublasCsyr(handle, CUBLAS_FILL_MODE_UPPER, n, alpha_c, x_c, incx, result_c, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, result_z_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr(*handle, oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, result_z_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZsyr(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, result_z, lda);
  cublasZsyr(handle, CUBLAS_FILL_MODE_UPPER, n, alpha_z, x_z, incx, result_z, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2(*handle, oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, incx, y_c_buf_ct{{[0-9]+}}, incy, result_c_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCsyr2(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2(*handle, oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, incx, y_z_buf_ct{{[0-9]+}}, incy, result_z_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZsyr2(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);


  // level 3
  int N = 100;
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(*handle, dpct::get_transpose(trans0), dpct::get_transpose(trans0), N, N, N, dpct::get_value(alpha_c, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), N, dpct::get_value(beta_c, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), N));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, dpct::get_value(alpha_c, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), N, dpct::get_value(beta_c, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), N);
  status = cublasCgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, N, N, N, alpha_c, x_c, N, y_c, N, beta_c, result_c, N);
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha_c, x_c, N, y_c, N, beta_c, result_c, N);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(*handle, dpct::get_transpose(trans0), dpct::get_transpose(trans0), N, N, N, dpct::get_value(alpha_z, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), N, dpct::get_value(beta_z, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), N));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, dpct::get_value(alpha_z, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), N, dpct::get_value(beta_z, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), N);
  status = cublasZgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, N, N, N, alpha_z, x_z, N, y_z, N, beta_z, result_z, N);
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha_z, x_z, N, y_z, N, beta_z, result_z, N);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, N, N, N, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, N, 16, y_c_buf_ct{{[0-9]+}}, N, 16, dpct::get_value(beta_c, *handle), result_c_buf_ct{{[0-9]+}}, N, 16, 10));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm_batch(*handle, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, trans1==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans1, N, N, N, dpct::get_value(alpha_c, *handle), x_c_buf_ct{{[0-9]+}}, N, 16, y_c_buf_ct{{[0-9]+}}, N, 16, dpct::get_value(beta_c, *handle), result_c_buf_ct{{[0-9]+}}, N, 16, 10);
  // CHECK-NEXT: }
  status = cublasCgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, alpha_c, x_c, N, 16, y_c, N, 16, beta_c, result_c, N, 16, 10);
  cublasCgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, alpha_c, x_c, N, 16, y_c, N, 16, beta_c, result_c, N, 16, 10);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(*handle, oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, N, 16, y_z_buf_ct{{[0-9]+}}, N, 16, dpct::get_value(beta_z, *handle), result_z_buf_ct{{[0-9]+}}, N, 16, 10));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm_batch(*handle, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, trans1==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans1, N, N, N, dpct::get_value(alpha_z, *handle), x_z_buf_ct{{[0-9]+}}, N, 16, y_z_buf_ct{{[0-9]+}}, N, 16, dpct::get_value(beta_z, *handle), result_z_buf_ct{{[0-9]+}}, N, 16, 10);
  // CHECK-NEXT: }
  status = cublasZgemmStridedBatched(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha_z, x_z, N, 16, y_z, N, 16, beta_z, result_z, N, 16, 10);
  cublasZgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, alpha_z, x_z, N, 16, y_z, N, 16, beta_z, result_z, N, 16, 10);

  const cuComplex** x_c_array = 0;
  const cuComplex** y_c_array = 0;
  cuComplex** result_c_array = 0;
  const cuDoubleComplex** x_z_array = 0;
  const cuDoubleComplex** y_z_array = 0;
  cuDoubleComplex** result_z_array = 0;
  cublasOperation_t trans3 = CUBLAS_OP_N;

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cublasCgemmBatched is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = cublasCgemmBatched(handle, trans3, trans3, N, N, N, alpha_c, x_c_array, N, y_c_array, N, beta_c, result_c_array, N, 10);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cublasCgemmBatched is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasCgemmBatched(handle, trans3, trans3, N, N, N, alpha_c, x_c_array, N, y_c_array, N, beta_c, result_c_array, N, 10);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cublasZgemmBatched is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = cublasZgemmBatched(handle, trans3, trans3, N, N, N, alpha_z, x_z_array, N, y_z_array, N, beta_z, result_z_array, N, 10);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cublasZgemmBatched is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasZgemmBatched(handle, trans3, trans3, N, N, N, alpha_z, x_z_array, N, y_z_array, N, beta_z, result_z_array, N, 10);
  status = cublasCgemmBatched(handle, trans3, trans3, N, N, N, alpha_c, x_c_array, N, y_c_array, N, beta_c, result_c_array, N, 10);
  cublasCgemmBatched(handle, trans3, trans3, N, N, N, alpha_c, x_c_array, N, y_c_array, N, beta_c, result_c_array, N, 10);
  status = cublasZgemmBatched(handle, trans3, trans3, N, N, N, alpha_z, x_z_array, N, y_z_array, N, beta_z, result_z_array, N, 10);
  cublasZgemmBatched(handle, trans3, trans3, N, N, N, alpha_z, x_z_array, N, y_z_array, N, beta_z, result_z_array, N, 10);

  cuComplex* A_c = 0;
  cuDoubleComplex* A_z = 0;
  cuComplex* B_c = 0;
  cuDoubleComplex* B_z = 0;
  cuComplex* C_c = 0;
  cuDoubleComplex* C_z = 0;


  int ldb = 10;
  int ldc = 10;


  const float alpha_s = 1;
  const float beta_s = 1;
  const double beta_d = 0;

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(*handle, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, m, n, k, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, B_c_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_c, *handle), C_c_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, m, n, k, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, B_c_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_c, *handle), C_c_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCgemm3m(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(*handle, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, m, n, k, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, B_z_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_z, *handle), C_z_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, m, n, k, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, B_z_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_z, *handle), C_z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZgemm3m(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  int side0 = 0;
  int side1 = 1;
  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(*handle, (oneapi::mkl::side)side0, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, m, n, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, B_c_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_c, *handle), C_c_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::lower, m, n, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, B_c_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_c, *handle), C_c_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCsymm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(*handle, (oneapi::mkl::side)side1, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, m, n, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, B_z_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_z, *handle), C_z_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::lower, m, n, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, B_z_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_z, *handle), C_z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZsymm(handle, (cublasSideMode_t)side1, (cublasFillMode_t)fill0, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, n, k, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, dpct::get_value(beta_c, *handle), C_c_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syrk(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, dpct::get_value(beta_c, *handle), C_c_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);
  cublasCsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, n, k, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, dpct::get_value(beta_z, *handle), C_z_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syrk(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, dpct::get_value(beta_z, *handle), C_z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);
  cublasZsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, n, k, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, B_c_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_c, *handle), C_c_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2k(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, B_c_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_c, *handle), C_c_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, n, k, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, B_z_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_z, *handle), C_z_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2k(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, B_z_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_z, *handle), C_z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(*handle, oneapi::mkl::side::left, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, m, n, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, B_c_buf_ct{{[0-9]+}}, ldb));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, B_c_buf_ct{{[0-9]+}}, ldb);
  // CHECK-NEXT: }
  status = cublasCtrsm(handle, (cublasSideMode_t)0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, alpha_c, A_c, lda, B_c, ldb);
  cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_c, A_c, lda, B_c, ldb);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(*handle, oneapi::mkl::side::right, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, m, n, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, B_z_buf_ct{{[0-9]+}}, ldb));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, B_z_buf_ct{{[0-9]+}}, ldb);
  // CHECK-NEXT: }
  status = cublasZtrsm(handle, (cublasSideMode_t)1, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, alpha_z, A_z, lda, B_z, ldb);
  cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_z, A_z, lda, B_z, ldb);

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemm(*handle, (oneapi::mkl::side)side0, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, m, n, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, B_c_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_c, *handle), C_c_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hemm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::lower, m, n, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, B_c_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_c, *handle), C_c_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasChemm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasChemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemm(*handle, (oneapi::mkl::side)side0, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, m, n, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, B_z_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_z, *handle), C_z_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hemm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::lower, m, n, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, B_z_buf_ct{{[0-9]+}}, ldb, dpct::get_value(beta_z, *handle), C_z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZhemm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZhemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::herk(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, n, k, alpha_s, A_c_buf_ct{{[0-9]+}}, lda, beta_s, C_c_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::herk(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, alpha_s, A_c_buf_ct{{[0-9]+}}, lda, beta_s, C_c_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCherk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_s, A_c, lda, &beta_s, C_c, ldc);
  cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &alpha_s, A_c, lda, &beta_s, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::herk(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, n, k, dpct::get_value(alpha_d, *handle), A_z_buf_ct{{[0-9]+}}, lda, beta_d, C_z_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::herk(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_d, *handle), A_z_buf_ct{{[0-9]+}}, lda, beta_d, C_z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZherk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_d, A_z, lda, &beta_d, C_z, ldc);
  cublasZherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_d, A_z, lda, &beta_d, C_z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2k(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, n, k, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, B_c_buf_ct{{[0-9]+}}, ldb, beta_s, C_c_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her2k(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, B_c_buf_ct{{[0-9]+}}, ldb, beta_s, C_c_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCher2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, B_c, ldb, &beta_s, C_c, ldc);
  cublasCher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, &beta_s, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2k(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, n, k, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, B_z_buf_ct{{[0-9]+}}, ldb, beta_d, C_z_buf_ct{{[0-9]+}}, ldc));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her2k(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, B_z_buf_ct{{[0-9]+}}, ldb, beta_d, C_z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZher2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, B_z, ldb, &beta_d, C_z, ldc);
  cublasZher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, &beta_d, C_z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto transpose_ct{{[0-9]+}} = foo();
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsm(*handle, (oneapi::mkl::side)foo(), foo()==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, (int)transpose_ct{{[0-9]+}}==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)transpose_ct{{[0-9]+}}, (oneapi::mkl::diag)foo(), m, n, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, B_z_buf_ct{{[0-9]+}}, ldb);
  // CHECK-NEXT: }
  cublasZtrsm(handle, (cublasSideMode_t)foo(), (cublasFillMode_t)foo(), (cublasOperation_t)foo(), (cublasDiagType_t)foo(), m, n, alpha_z, A_z, lda, B_z, ldb);

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::omatadd(*handle, oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::trans, m, n, dpct::get_value(alpha_c, *handle), A_c_buf_ct{{[0-9]+}}, lda, dpct::get_value(beta_c, *handle), B_c_buf_ct{{[0-9]+}}, ldb, C_c_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCgeam(handle, CUBLAS_OP_C, CUBLAS_OP_T, m, n, alpha_c, A_c, lda, beta_c, B_c, ldb, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::omatadd(*handle, oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::trans, m, n, dpct::get_value(alpha_z, *handle), A_z_buf_ct{{[0-9]+}}, lda, dpct::get_value(beta_z, *handle), B_z_buf_ct{{[0-9]+}}, ldb, C_z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_T, m, n, alpha_z, A_z, lda, beta_z, B_z, ldb, C_z, ldc);
}

