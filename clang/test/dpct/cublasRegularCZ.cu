// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasRegularCZ.dp.cpp --match-full-lines %s

#include <cuda_runtime.h>
#include <cublas_v2.h>

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
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_buf_ct1 = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> result_temp_buffer(sycl::range<1>(1));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::iamax(handle, n, x_c_buf_ct1, incx, result_temp_buffer), 0);
  // CHECK-NEXT: result_buf_ct1.get_access<sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_buf_ct1 = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> result_temp_buffer(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamax(handle, n, x_c_buf_ct1, incx, result_temp_buffer);
  // CHECK-NEXT: result_buf_ct1.get_access<sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIcamax(handle, n, x_c, incx, result);
  cublasIcamax(handle, n, x_c, incx, result);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_buf_ct1 = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> result_temp_buffer(sycl::range<1>(1));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::iamax(handle, n, x_z_buf_ct1, incx, result_temp_buffer), 0);
  // CHECK-NEXT: result_buf_ct1.get_access<sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_buf_ct1 = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> result_temp_buffer(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamax(handle, n, x_z_buf_ct1, incx, result_temp_buffer);
  // CHECK-NEXT: result_buf_ct1.get_access<sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIzamax(handle, n, x_z, incx, result);
  cublasIzamax(handle, n, x_z, incx, result);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_buf_ct1 = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> result_temp_buffer(sycl::range<1>(1));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::iamin(handle, n, x_c_buf_ct1, incx, result_temp_buffer), 0);
  // CHECK-NEXT: result_buf_ct1.get_access<sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_buf_ct1 = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> result_temp_buffer(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamin(handle, n, x_c_buf_ct1, incx, result_temp_buffer);
  // CHECK-NEXT: result_buf_ct1.get_access<sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIcamin(handle, n, x_c, incx, result);
  cublasIcamin(handle, n, x_c, incx, result);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_buf_ct1 = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> result_temp_buffer(sycl::range<1>(1));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::iamin(handle, n, x_z_buf_ct1, incx, result_temp_buffer), 0);
  // CHECK-NEXT: result_buf_ct1.get_access<sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_buf_ct1 = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> result_temp_buffer(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamin(handle, n, x_z_buf_ct1, incx, result_temp_buffer);
  // CHECK-NEXT: result_buf_ct1.get_access<sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIzamin(handle, n, x_z, incx, result);
  cublasIzamin(handle, n, x_z, incx, result);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_f_buf_ct1 = dpct::get_buffer<float>(result_f);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::asum(handle, n, x_c_buf_ct1, incx, result_f_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_f_buf_ct1 = dpct::get_buffer<float>(result_f);
  // CHECK-NEXT: mkl::blas::asum(handle, n, x_c_buf_ct1, incx, result_f_buf_ct1);
  // CHECK-NEXT: }
  status = cublasScasum(handle, n, x_c, incx, result_f);
  cublasScasum(handle, n, x_c, incx, result_f);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_d_buf_ct1 = dpct::get_buffer<double>(result_d);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::asum(handle, n, x_z_buf_ct1, incx, result_d_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_d_buf_ct1 = dpct::get_buffer<double>(result_d);
  // CHECK-NEXT: mkl::blas::asum(handle, n, x_z_buf_ct1, incx, result_d_buf_ct1);
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
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::axpy(handle, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, incx, y_c_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: mkl::blas::axpy(handle, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, incx, y_c_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasCaxpy(handle, n, alpha_c, x_c, incx, y_c, incy);
  cublasCaxpy(handle, n, alpha_c, x_c, incx, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::axpy(handle, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, incx, y_z_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: mkl::blas::axpy(handle, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, incx, y_z_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasZaxpy(handle, n, alpha_z, x_z, incx, y_z, incy);
  cublasZaxpy(handle, n, alpha_z, x_z, incx, y_z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::copy(handle, n, x_c_buf_ct1, incx, y_c_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: mkl::blas::copy(handle, n, x_c_buf_ct1, incx, y_c_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasCcopy(handle, n, x_c, incx, y_c, incy);
  cublasCcopy(handle, n, x_c, incx, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::copy(handle, n, x_z_buf_ct1, incx, y_z_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: mkl::blas::copy(handle, n, x_z_buf_ct1, incx, y_z_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasZcopy(handle, n, x_z, incx, y_z, incy);
  cublasZcopy(handle, n, x_z, incx, y_z, incy);

  cuComplex* result_c = 0;
  cuDoubleComplex* result_z = 0;

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::dotu(handle, n, x_c_buf_ct1, incx, y_c_buf_ct1, incy, result_c_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::dotu(handle, n, x_c_buf_ct1, incx, y_c_buf_ct1, incy, result_c_buf_ct1);
  // CHECK-NEXT: }
  status = cublasCdotu(handle, n, x_c, incx, y_c, incy, result_c);
  cublasCdotu(handle, n, x_c, incx, y_c, incy, result_c);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::dotc(handle, n, x_c_buf_ct1, incx, y_c_buf_ct1, incy, result_c_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::dotc(handle, n, x_c_buf_ct1, incx, y_c_buf_ct1, incy, result_c_buf_ct1);
  // CHECK-NEXT: }
  status = cublasCdotc(handle, n, x_c, incx, y_c, incy, result_c);
  cublasCdotc(handle, n, x_c, incx, y_c, incy, result_c);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::dotu(handle, n, x_z_buf_ct1, incx, y_z_buf_ct1, incy, result_z_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::dotu(handle, n, x_z_buf_ct1, incx, y_z_buf_ct1, incy, result_z_buf_ct1);
  // CHECK-NEXT: }
  status = cublasZdotu(handle, n, x_z, incx, y_z, incy, result_z);
  cublasZdotu(handle, n, x_z, incx, y_z, incy, result_z);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::dotc(handle, n, x_z_buf_ct1, incx, y_z_buf_ct1, incy, result_z_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::dotc(handle, n, x_z_buf_ct1, incx, y_z_buf_ct1, incy, result_z_buf_ct1);
  // CHECK-NEXT: }
  status = cublasZdotc(handle, n, x_z, incx, y_z, incy, result_z);
  cublasZdotc(handle, n, x_z, incx, y_z, incy, result_z);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_f_buf_ct1 = dpct::get_buffer<float>(result_f);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::nrm2(handle, n, x_c_buf_ct1, incx, result_f_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_f_buf_ct1 = dpct::get_buffer<float>(result_f);
  // CHECK-NEXT: mkl::blas::nrm2(handle, n, x_c_buf_ct1, incx, result_f_buf_ct1);
  // CHECK-NEXT: }
  status = cublasScnrm2(handle, n, x_c, incx, result_f);
  cublasScnrm2(handle, n, x_c, incx, result_f);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_d_buf_ct1 = dpct::get_buffer<double>(result_d);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::nrm2(handle, n, x_z_buf_ct1, incx, result_d_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_d_buf_ct1 = dpct::get_buffer<double>(result_d);
  // CHECK-NEXT: mkl::blas::nrm2(handle, n, x_z_buf_ct1, incx, result_d_buf_ct1);
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
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::rot(handle, n, x_c_buf_ct1, incx, y_c_buf_ct1, incy, *(c_f), *(s_f)), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: mkl::blas::rot(handle, n, x_c_buf_ct1, incx, y_c_buf_ct1, incy, *(c_f), *(s_f));
  // CHECK-NEXT: }
  status = cublasCsrot(handle, n, x_c, incx, y_c, incy, c_f, s_f);
  cublasCsrot(handle, n, x_c, incx, y_c, incy, c_f, s_f);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::rot(handle, n, x_z_buf_ct1, incx, y_z_buf_ct1, incy, *(c_d), *(s_d)), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: mkl::blas::rot(handle, n, x_z_buf_ct1, incx, y_z_buf_ct1, incy, *(c_d), *(s_d));
  // CHECK-NEXT: }
  status = cublasZdrot(handle, n, x_z, incx, y_z, incy, c_d, s_d);
  cublasZdrot(handle, n, x_z, incx, y_z, incy, c_d, s_d);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto c_f_buf_ct1 = dpct::get_buffer<float>(c_f);
  // CHECK-NEXT: auto s_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(s_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::rotg(handle, x_c_buf_ct1, y_c_buf_ct1, c_f_buf_ct1, s_c_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto c_f_buf_ct1 = dpct::get_buffer<float>(c_f);
  // CHECK-NEXT: auto s_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(s_c);
  // CHECK-NEXT: mkl::blas::rotg(handle, x_c_buf_ct1, y_c_buf_ct1, c_f_buf_ct1, s_c_buf_ct1);
  // CHECK-NEXT: }
  status = cublasCrotg(handle, x_c, y_c, c_f, s_c);
  cublasCrotg(handle, x_c, y_c, c_f, s_c);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto c_d_buf_ct1 = dpct::get_buffer<double>(c_d);
  // CHECK-NEXT: auto s_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(s_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::rotg(handle, x_z_buf_ct1, y_z_buf_ct1, c_d_buf_ct1, s_z_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto c_d_buf_ct1 = dpct::get_buffer<double>(c_d);
  // CHECK-NEXT: auto s_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(s_z);
  // CHECK-NEXT: mkl::blas::rotg(handle, x_z_buf_ct1, y_z_buf_ct1, c_d_buf_ct1, s_z_buf_ct1);
  // CHECK-NEXT: }
  status = cublasZrotg(handle, x_z, y_z, c_d, s_z);
  cublasZrotg(handle, x_z, y_z, c_d, s_z);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::scal(handle, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: mkl::blas::scal(handle, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasCscal(handle, n, alpha_c, x_c, incx);
  cublasCscal(handle, n, alpha_c, x_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::scal(handle, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: mkl::blas::scal(handle, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasZscal(handle, n, alpha_z, x_z, incx);
  cublasZscal(handle, n, alpha_z, x_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::scal(handle, n, *(alpha_f), x_c_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: mkl::blas::scal(handle, n, *(alpha_f), x_c_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasCsscal(handle, n, alpha_f, x_c, incx);
  cublasCsscal(handle, n, alpha_f, x_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::scal(handle, n, *(alpha_d), x_z_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: mkl::blas::scal(handle, n, *(alpha_d), x_z_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasZdscal(handle, n, alpha_d, x_z, incx);
  cublasZdscal(handle, n, alpha_d, x_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::swap(handle, n, x_c_buf_ct1, incx, y_c_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: mkl::blas::swap(handle, n, x_c_buf_ct1, incx, y_c_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasCswap(handle, n, x_c, incx, y_c, incy);
  cublasCswap(handle, n, x_c, incx, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::swap(handle, n, x_z_buf_ct1, incx, y_z_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: mkl::blas::swap(handle, n, x_z_buf_ct1, incx, y_z_buf_ct1, incy);
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
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct2 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gbmv(handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, m, n, kl, ku, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, lda, x_c_buf_ct2, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct2 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: mkl::blas::gbmv(handle, mkl::transpose::nontrans, m, n, kl, ku, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, lda, x_c_buf_ct2, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasCgbmv(handle, (cublasOperation_t)trans0, m, n, kl, ku, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasCgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct2 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gbmv(handle, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, m, n, kl, ku, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, lda, x_z_buf_ct2, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct2 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: mkl::blas::gbmv(handle, mkl::transpose::nontrans, m, n, kl, ku, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, lda, x_z_buf_ct2, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasZgbmv(handle, (cublasOperation_t)trans1, m, n, kl, ku, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct2 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gemv(handle, trans2==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans2, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, lda, x_c_buf_ct2, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct2 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: mkl::blas::gemv(handle, mkl::transpose::nontrans, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, lda, x_c_buf_ct2, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasCgemv(handle, (cublasOperation_t)trans2, m, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasCgemv(handle, CUBLAS_OP_N, m, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct2 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gemv(handle, mkl::transpose::nontrans, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, lda, x_z_buf_ct2, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct2 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: mkl::blas::gemv(handle, mkl::transpose::nontrans, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, lda, x_z_buf_ct2, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasZgemv(handle, (cublasOperation_t)0, m, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZgemv(handle, CUBLAS_OP_N, m, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::geru(handle, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, incx, y_c_buf_ct1, incy, result_c_buf_ct1, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::geru(handle, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, incx, y_c_buf_ct1, incy, result_c_buf_ct1, lda);
  // CHECK-NEXT: }
  status = cublasCgeru(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCgeru(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gerc(handle, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, incx, y_c_buf_ct1, incy, result_c_buf_ct1, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::gerc(handle, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, incx, y_c_buf_ct1, incy, result_c_buf_ct1, lda);
  // CHECK-NEXT: }
  status = cublasCgerc(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCgerc(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::geru(handle, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, incx, y_z_buf_ct1, incy, result_z_buf_ct1, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::geru(handle, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, incx, y_z_buf_ct1, incy, result_z_buf_ct1, lda);
  // CHECK-NEXT: }
  status = cublasZgeru(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZgeru(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gerc(handle, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, incx, y_z_buf_ct1, incy, result_z_buf_ct1, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::gerc(handle, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, incx, y_z_buf_ct1, incy, result_z_buf_ct1, lda);
  // CHECK-NEXT: }
  status = cublasZgerc(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZgerc(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  int k = 1;
  int fill0 = 0;
  int fill1 = 1;
  int diag0 = 0;
  int diag1 = 1;
  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tbmv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, mkl::transpose::trans, (mkl::diag)diag0, n, k, x_c_buf_ct1, lda, result_c_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::tbmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, k, x_c_buf_ct1, lda, result_c_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasCtbmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)1, (cublasDiagType_t)diag0, n, k, x_c, lda, result_c, incx);
  cublasCtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_c, lda, result_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tbmv(handle, fill1==0 ? mkl::uplo::lower : mkl::uplo::upper, mkl::transpose::conjtrans, (mkl::diag)diag1, n, k, x_z_buf_ct1, lda, result_z_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::tbmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, k, x_z_buf_ct1, lda, result_z_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasZtbmv(handle, (cublasFillMode_t)fill1, (cublasOperation_t)2, (cublasDiagType_t)diag1, n, k, x_z, lda, result_z, incx);
  cublasZtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_z, lda, result_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tbsv(handle, mkl::uplo::lower, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, mkl::diag::nonunit,  n, k, x_c_buf_ct1, lda, result_c_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::tbsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit,  n, k, x_c_buf_ct1, lda, result_c_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasCtbsv(handle, (cublasFillMode_t)0, (cublasOperation_t)trans0, (cublasDiagType_t)0,  n, k, x_c, lda, result_c, incx);
  cublasCtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_c, lda, result_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tbsv(handle, mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, mkl::diag::unit,  n, k, x_z_buf_ct1, lda, result_z_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::tbsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit,  n, k, x_z_buf_ct1, lda, result_z_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasZtbsv(handle, (cublasFillMode_t)1, (cublasOperation_t)trans0, (cublasDiagType_t)1,  n, k, x_z, lda, result_z, incx);
  cublasZtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_z, lda, result_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tpmv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_c_buf_ct1, result_c_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::tpmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_c_buf_ct1, result_c_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasCtpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, result_c, incx);
  cublasCtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tpmv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_z_buf_ct1, result_z_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::tpmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_z_buf_ct1, result_z_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasZtpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, result_z, incx);
  cublasZtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tpsv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_c_buf_ct1, result_c_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::tpsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_c_buf_ct1, result_c_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasCtpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, result_c, incx);
  cublasCtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tpsv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_z_buf_ct1, result_z_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::tpsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_z_buf_ct1, result_z_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasZtpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, result_z, incx);
  cublasZtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::trmv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_c_buf_ct1, lda, result_c_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::trmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_c_buf_ct1, lda, result_c_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasCtrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, lda, result_c, incx);
  cublasCtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::trmv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_z_buf_ct1, lda, result_z_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::trmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_z_buf_ct1, lda, result_z_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasZtrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, lda, result_z, incx);
  cublasZtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::trsv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_c_buf_ct1, lda, result_c_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::trsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_c_buf_ct1, lda, result_c_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasCtrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, lda, result_c, incx);
  cublasCtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::trsv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_z_buf_ct1, lda, result_z_buf_ct1, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::trsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_z_buf_ct1, lda, result_z_buf_ct1, incx);
  // CHECK-NEXT: }
  status = cublasZtrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, lda, result_z, incx);
  cublasZtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct2 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::hemv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, lda, x_c_buf_ct2, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct2 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: mkl::blas::hemv(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, lda, x_c_buf_ct2, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasChemv(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasChemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct2 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::hemv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, lda, x_z_buf_ct2, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct2 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: mkl::blas::hemv(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, lda, x_z_buf_ct2, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasZhemv(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZhemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct2 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::hbmv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, lda, x_c_buf_ct2, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct2 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: mkl::blas::hbmv(handle, mkl::uplo::lower, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, lda, x_c_buf_ct2, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasChbmv(handle, (cublasFillMode_t)fill0, n, k, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasChbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct2 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::hbmv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, lda, x_z_buf_ct2, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct2 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: mkl::blas::hbmv(handle, mkl::uplo::lower, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, lda, x_z_buf_ct2, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasZhbmv(handle, (cublasFillMode_t)fill0, n, k, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZhbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct2 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::hpmv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, x_c_buf_ct2, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto x_c_buf_ct2 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: mkl::blas::hpmv(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, x_c_buf_ct2, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasChpmv(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, x_c, incx, beta_c, y_c, incy);
  cublasChpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, x_c, incx, beta_c, y_c, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct2 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::hpmv(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, x_z_buf_ct2, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_buf_ct1, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto x_z_buf_ct2 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: mkl::blas::hpmv(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, x_z_buf_ct2, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_buf_ct1, incy);
  // CHECK-NEXT: }
  status = cublasZhpmv(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, x_z, incx, beta_z, y_z, incy);
  cublasZhpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, x_z, incx, beta_z, y_z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::her(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, *(alpha_f), x_c_buf_ct1, incx, result_c_buf_ct1, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::her(handle, mkl::uplo::lower, n, *(alpha_f), x_c_buf_ct1, incx, result_c_buf_ct1, lda);
  // CHECK-NEXT: }
  status = cublasCher(handle, (cublasFillMode_t)fill0, n, alpha_f, x_c, incx, result_c, lda);
  cublasCher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::her(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, *(alpha_d), x_z_buf_ct1, incx, result_z_buf_ct1, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::her(handle, mkl::uplo::lower, n, *(alpha_d), x_z_buf_ct1, incx, result_z_buf_ct1, lda);
  // CHECK-NEXT: }
  status = cublasZher(handle, (cublasFillMode_t)fill0, n, alpha_d, x_z, incx, result_z, lda);
  cublasZher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::her2(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, incx, y_c_buf_ct1, incy, result_c_buf_ct1, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::her2(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, incx, y_c_buf_ct1, incy, result_c_buf_ct1, lda);
  // CHECK-NEXT: }
  status = cublasCher2(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::her2(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, incx, y_z_buf_ct1, incy, result_z_buf_ct1, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::her2(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, incx, y_z_buf_ct1, incy, result_z_buf_ct1, lda);
  // CHECK-NEXT: }
  status = cublasZher2(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::hpr(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, *(alpha_f), x_c_buf_ct1, incx, result_c_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::hpr(handle, mkl::uplo::lower, n, *(alpha_f), x_c_buf_ct1, incx, result_c_buf_ct1);
  // CHECK-NEXT: }
  status = cublasChpr(handle, (cublasFillMode_t)fill0, n, alpha_f, x_c, incx, result_c);
  cublasChpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::hpr(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, *(alpha_d), x_z_buf_ct1, incx, result_z_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::hpr(handle, mkl::uplo::lower, n, *(alpha_d), x_z_buf_ct1, incx, result_z_buf_ct1);
  // CHECK-NEXT: }
  status = cublasZhpr(handle, (cublasFillMode_t)fill0, n, alpha_d, x_z, incx, result_z);
  cublasZhpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z);

  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::hpr2(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, incx, y_c_buf_ct1, incy, result_c_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::hpr2(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, incx, y_c_buf_ct1, incy, result_c_buf_ct1);
  // CHECK-NEXT: }
  status = cublasChpr2(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, y_c, incy, result_c);
  cublasChpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::hpr2(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, incx, y_z_buf_ct1, incy, result_z_buf_ct1), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::hpr2(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, incx, y_z_buf_ct1, incy, result_z_buf_ct1);
  // CHECK-NEXT: }
  status = cublasZhpr2(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, y_z, incy, result_z);
  cublasZhpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z);

  int N = 100;
  // CHECK: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gemm(handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, N, N, N, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, N, y_c_buf_ct1, N, std::complex<float>((beta_c)->x(),(beta_c)->y()), result_c_buf_ct1, N), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(x_c);
  // CHECK-NEXT: auto y_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(y_c);
  // CHECK-NEXT: auto result_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(result_c);
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_buf_ct1, N, y_c_buf_ct1, N, std::complex<float>((beta_c)->x(),(beta_c)->y()), result_c_buf_ct1, N);
  // CHECK-NEXT: }
  status = cublasCgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, N, N, N, alpha_c, x_c, N, y_c, N, beta_c, result_c, N);
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha_c, x_c, N, y_c, N, beta_c, result_c, N);

  // CHECK: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gemm(handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, N, N, N, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, N, y_z_buf_ct1, N, std::complex<double>((beta_z)->x(),(beta_z)->y()), result_z_buf_ct1, N), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(x_z);
  // CHECK-NEXT: auto y_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(y_z);
  // CHECK-NEXT: auto result_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(result_z);
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_buf_ct1, N, y_z_buf_ct1, N, std::complex<double>((beta_z)->x(),(beta_z)->y()), result_z_buf_ct1, N);
  // CHECK-NEXT: }
  status = cublasZgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, N, N, N, alpha_z, x_z, N, y_z, N, beta_z, result_z, N);
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha_z, x_z, N, y_z, N, beta_z, result_z, N);

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
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::cgemm3m(handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, m, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, B_c_buf_ct1, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: mkl::blas::cgemm3m(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, m, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, B_c_buf_ct1, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasCgemm3m(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::zgemm3m(handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, m, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, B_z_buf_ct1, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: mkl::blas::zgemm3m(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, m, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, B_z_buf_ct1, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasZgemm3m(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  int side0 = 0;
  int side1 = 1;
  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::symm(handle, (mkl::side)side0, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, B_c_buf_ct1, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: mkl::blas::symm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, B_c_buf_ct1, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasCsymm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::symm(handle, (mkl::side)side1, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, B_z_buf_ct1, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: mkl::blas::symm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, B_z_buf_ct1, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasZsymm(handle, (cublasSideMode_t)side1, (cublasFillMode_t)fill0, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::syrk(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: mkl::blas::syrk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasCsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);
  cublasCsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::syrk(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: mkl::blas::syrk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasZsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);
  cublasZsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::syr2k(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, B_c_buf_ct1, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: mkl::blas::syr2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, B_c_buf_ct1, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasCsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::syr2k(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, B_z_buf_ct1, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: mkl::blas::syr2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, B_z_buf_ct1, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasZsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::trsm(handle, mkl::side::left, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, B_c_buf_ct1, ldb), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: mkl::blas::trsm(handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, B_c_buf_ct1, ldb);
  // CHECK-NEXT: }
  status = cublasCtrsm(handle, (cublasSideMode_t)0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, alpha_c, A_c, lda, B_c, ldb);
  cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_c, A_c, lda, B_c, ldb);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::trsm(handle, mkl::side::right, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, B_z_buf_ct1, ldb), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: mkl::blas::trsm(handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, B_z_buf_ct1, ldb);
  // CHECK-NEXT: }
  status = cublasZtrsm(handle, (cublasSideMode_t)1, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, alpha_z, A_z, lda, B_z, ldb);
  cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_z, A_z, lda, B_z, ldb);

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::hemm(handle, (mkl::side)side0, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, B_c_buf_ct1, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: mkl::blas::hemm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, B_c_buf_ct1, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasChemm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasChemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::hemm(handle, (mkl::side)side0, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, B_z_buf_ct1, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: mkl::blas::hemm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, B_z_buf_ct1, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasZhemm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZhemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::herk(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, n, k, alpha_s, A_c_buf_ct1, lda, beta_s, C_c_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: mkl::blas::herk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, alpha_s, A_c_buf_ct1, lda, beta_s, C_c_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasCherk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_s, A_c, lda, &beta_s, C_c, ldc);
  cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &alpha_s, A_c, lda, &beta_s, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::herk(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, n, k, *(alpha_d), A_z_buf_ct1, lda, beta_d, C_z_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: mkl::blas::herk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, *(alpha_d), A_z_buf_ct1, lda, beta_d, C_z_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasZherk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_d, A_z, lda, &beta_d, C_z, ldc);
  cublasZherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_d, A_z, lda, &beta_d, C_z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::her2k(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, B_c_buf_ct1, ldb, beta_s, C_c_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(A_c);
  // CHECK-NEXT: auto B_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(B_c);
  // CHECK-NEXT: auto C_c_buf_ct1 = dpct::get_buffer<std::complex<float>>(C_c);
  // CHECK-NEXT: mkl::blas::her2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_buf_ct1, lda, B_c_buf_ct1, ldb, beta_s, C_c_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasCher2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, B_c, ldb, &beta_s, C_c, ldc);
  cublasCher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, &beta_s, C_c, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::her2k(handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, B_z_buf_ct1, ldb, beta_d, C_z_buf_ct1, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(A_z);
  // CHECK-NEXT: auto B_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(B_z);
  // CHECK-NEXT: auto C_z_buf_ct1 = dpct::get_buffer<std::complex<double>>(C_z);
  // CHECK-NEXT: mkl::blas::her2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_buf_ct1, lda, B_z_buf_ct1, ldb, beta_d, C_z_buf_ct1, ldc);
  // CHECK-NEXT: }
  status = cublasZher2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, B_z, ldb, &beta_d, C_z, ldc);
  cublasZher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, &beta_d, C_z, ldc);
}
