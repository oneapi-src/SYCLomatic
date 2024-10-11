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
  // CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasIcamax(handle, n, x_c, incx, result);
  cublasIcamax(handle, n, x_c, incx, result);

  // CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasIzamax(handle, n, x_z, incx, result);
  cublasIzamax(handle, n, x_z, incx, result);

  // CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasIcamin(handle, n, x_c, incx, result);
  cublasIcamin(handle, n, x_c, incx, result);

  // CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasIzamin(handle, n, x_z, incx, result);
  cublasIzamin(handle, n, x_z, incx, result);

  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), result_f);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), result_f);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasScasum(handle, n, x_c, incx, result_f);
  cublasScasum(handle, n, x_c, incx, result_f);

  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), result_d);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), result_d);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
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

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy);
  status = cublasCaxpy(handle, n, alpha_c, x_c, incx, y_c, incy);
  cublasCaxpy(handle, n, alpha_c, x_c, incx, y_c, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy);
  status = cublasZaxpy(handle, n, alpha_z, x_z, incx, y_z, incy);
  cublasZaxpy(handle, n, alpha_z, x_z, incx, y_z, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy);
  status = cublasCcopy(handle, n, x_c, incx, y_c, incy);
  cublasCcopy(handle, n, x_c, incx, y_c, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy);
  status = cublasZcopy(handle, n, x_z, incx, y_z, incy);
  cublasZcopy(handle, n, x_z, incx, y_z, incy);

  cuComplex* result_c = 0;
  cuDoubleComplex* result_z = 0;

  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float2_out res_wrapper_ct6(handle->get_queue(), result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotu(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float2_out res_wrapper_ct6(handle->get_queue(), result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotu(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasCdotu(handle, n, x_c, incx, y_c, incy, result_c);
  cublasCdotu(handle, n, x_c, incx, y_c, incy, result_c);

  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float2_out res_wrapper_ct6(handle->get_queue(), result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotc(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float2_out res_wrapper_ct6(handle->get_queue(), result_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotc(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasCdotc(handle, n, x_c, incx, y_c, incy, result_c);
  cublasCdotc(handle, n, x_c, incx, y_c, incy, result_c);

  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double2_out res_wrapper_ct6(handle->get_queue(), result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotu(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double2_out res_wrapper_ct6(handle->get_queue(), result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotu(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasZdotu(handle, n, x_z, incx, y_z, incy, result_z);
  cublasZdotu(handle, n, x_z, incx, y_z, incy, result_z);

  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double2_out res_wrapper_ct6(handle->get_queue(), result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotc(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double2_out res_wrapper_ct6(handle->get_queue(), result_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dotc(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(res_wrapper_ct6.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasZdotc(handle, n, x_z, incx, y_z, incy, result_z);
  cublasZdotc(handle, n, x_z, incx, y_z, incy, result_z);

  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), result_f);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), result_f);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasScnrm2(handle, n, x_c, incx, result_f);
  cublasScnrm2(handle, n, x_c, incx, result_f);

  // CHECK: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), result_d);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), result_d);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
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

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::get_value(c_f, handle->get_queue()), dpct::get_value(s_f, handle->get_queue())));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::get_value(c_f, handle->get_queue()), dpct::get_value(s_f, handle->get_queue()));
  status = cublasCsrot(handle, n, x_c, incx, y_c, incy, c_f, s_f);
  cublasCsrot(handle, n, x_c, incx, y_c, incy, c_f, s_f);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::get_value(c_d, handle->get_queue()), dpct::get_value(s_d, handle->get_queue())));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::get_value(c_d, handle->get_queue()), dpct::get_value(s_d, handle->get_queue()));
  status = cublasZdrot(handle, n, x_z, incx, y_z, incy, c_d, s_d);
  cublasZdrot(handle, n, x_z, incx, y_z, incy, c_d, s_d);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::get_value(c_f, handle->get_queue()), dpct::get_value(s_c, handle->get_queue())));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::get_value(c_f, handle->get_queue()), dpct::get_value(s_c, handle->get_queue()));
  status = cublasCrot(handle, n, x_c, incx, y_c, incy, c_f, s_c);
  cublasCrot(handle, n, x_c, incx, y_c, incy, c_f, s_c);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::get_value(c_d, handle->get_queue()), dpct::get_value(s_z, handle->get_queue())));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::get_value(c_d, handle->get_queue()), dpct::get_value(s_z, handle->get_queue()));
  status = cublasZrot(handle, n, x_z, incx, y_z, incy, c_d, s_z);
  cublasZrot(handle, n, x_z, incx, y_z, incy, c_d, s_z);

  // CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float2_inout res_wrapper_ct1(handle->get_queue(), x_c);
  // CHECK-NEXT: dpct::blas::wrapper_float2_inout res_wrapper_ct2(handle->get_queue(), y_c);
  // CHECK-NEXT: dpct::blas::wrapper_float_out res_wrapper_ct3(handle->get_queue(), c_f);
  // CHECK-NEXT: dpct::blas::wrapper_float2_out res_wrapper_ct4(handle->get_queue(), s_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotg(handle->get_queue(), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(res_wrapper_ct1.get_ptr())), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(res_wrapper_ct2.get_ptr())), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(res_wrapper_ct3.get_ptr())), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_float2_inout res_wrapper_ct1(handle->get_queue(), x_c);
  // CHECK-NEXT: dpct::blas::wrapper_float2_inout res_wrapper_ct2(handle->get_queue(), y_c);
  // CHECK-NEXT: dpct::blas::wrapper_float_out res_wrapper_ct3(handle->get_queue(), c_f);
  // CHECK-NEXT: dpct::blas::wrapper_float2_out res_wrapper_ct4(handle->get_queue(), s_c);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotg(handle->get_queue(), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(res_wrapper_ct1.get_ptr())), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(res_wrapper_ct2.get_ptr())), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(res_wrapper_ct3.get_ptr())), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasCrotg(handle, x_c, y_c, c_f, s_c);
  cublasCrotg(handle, x_c, y_c, c_f, s_c);

  // CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double2_inout res_wrapper_ct1(handle->get_queue(), x_z);
  // CHECK-NEXT: dpct::blas::wrapper_double2_inout res_wrapper_ct2(handle->get_queue(), y_z);
  // CHECK-NEXT: dpct::blas::wrapper_double_out res_wrapper_ct3(handle->get_queue(), c_d);
  // CHECK-NEXT: dpct::blas::wrapper_double2_out res_wrapper_ct4(handle->get_queue(), s_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotg(handle->get_queue(), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(res_wrapper_ct1.get_ptr())), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(res_wrapper_ct2.get_ptr())), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(res_wrapper_ct3.get_ptr())), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_double2_inout res_wrapper_ct1(handle->get_queue(), x_z);
  // CHECK-NEXT: dpct::blas::wrapper_double2_inout res_wrapper_ct2(handle->get_queue(), y_z);
  // CHECK-NEXT: dpct::blas::wrapper_double_out res_wrapper_ct3(handle->get_queue(), c_d);
  // CHECK-NEXT: dpct::blas::wrapper_double2_out res_wrapper_ct4(handle->get_queue(), s_z);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotg(handle->get_queue(), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(res_wrapper_ct1.get_ptr())), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(res_wrapper_ct2.get_ptr())), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(res_wrapper_ct3.get_ptr())), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(res_wrapper_ct4.get_ptr())));
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasZrotg(handle, x_z, y_z, c_d, s_z);
  cublasZrotg(handle, x_z, y_z, c_d, s_z);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx);
  status = cublasCscal(handle, n, alpha_c, x_c, incx);
  cublasCscal(handle, n, alpha_c, x_c, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx);
  status = cublasZscal(handle, n, alpha_z, x_z, incx);
  cublasZscal(handle, n, alpha_z, x_z, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_f, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_f, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx);
  status = cublasCsscal(handle, n, alpha_f, x_c, incx);
  cublasCsscal(handle, n, alpha_f, x_c, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx);
  status = cublasZdscal(handle, n, alpha_d, x_z, incx);
  cublasZdscal(handle, n, alpha_d, x_z, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy);
  status = cublasCswap(handle, n, x_c, incx, y_c, incy);
  cublasCswap(handle, n, x_c, incx, y_c, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy);
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
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), dpct::get_transpose(trans0), m, n, kl, ku, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), oneapi::mkl::transpose::nontrans, m, n, kl, ku, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy);
  status = cublasCgbmv(handle, (cublasOperation_t)trans0, m, n, kl, ku, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasCgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), dpct::get_transpose(trans1), m, n, kl, ku, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), oneapi::mkl::transpose::nontrans, m, n, kl, ku, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy);
  status = cublasZgbmv(handle, (cublasOperation_t)trans1, m, n, kl, ku, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemv(handle->get_queue(), dpct::get_transpose(trans2), m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemv(handle->get_queue(), oneapi::mkl::transpose::nontrans, m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy);
  status = cublasCgemv(handle, (cublasOperation_t)trans2, m, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasCgemv(handle, CUBLAS_OP_N, m, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemv(handle->get_queue(), oneapi::mkl::transpose::nontrans, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemv(handle->get_queue(), oneapi::mkl::transpose::nontrans, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy);
  status = cublasZgemv(handle, (cublasOperation_t)0, m, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZgemv(handle, CUBLAS_OP_N, m, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::geru(handle->get_queue(), m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::geru(handle->get_queue(), m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), lda);
  status = cublasCgeru(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCgeru(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gerc(handle->get_queue(), m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gerc(handle->get_queue(), m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), lda);
  status = cublasCgerc(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCgerc(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::geru(handle->get_queue(), m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::geru(handle->get_queue(), m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), lda);
  status = cublasZgeru(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZgeru(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gerc(handle->get_queue(), m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gerc(handle->get_queue(), m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), lda);
  status = cublasZgerc(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZgerc(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  int k = 1;
  int fill0 = 0;
  int fill1 = 1;
  int diag0 = 0;
  int diag1 = 1;
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, (oneapi::mkl::diag)diag0, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incx);
  status = cublasCtbmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)1, (cublasDiagType_t)diag0, n, k, x_c, lda, result_c, incx);
  cublasCtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_c, lda, result_c, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), fill1 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, (oneapi::mkl::diag)diag1, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incx);
  status = cublasZtbmv(handle, (cublasFillMode_t)fill1, (cublasOperation_t)2, (cublasDiagType_t)diag1, n, k, x_z, lda, result_z, incx);
  cublasZtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_z, lda, result_z, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), oneapi::mkl::uplo::lower, dpct::get_transpose(trans0), oneapi::mkl::diag::nonunit, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incx);
  status = cublasCtbsv(handle, (cublasFillMode_t)0, (cublasOperation_t)trans0, (cublasDiagType_t)0,  n, k, x_c, lda, result_c, incx);
  cublasCtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_c, lda, result_c, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), oneapi::mkl::diag::unit, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, k, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incx);
  status = cublasZtbsv(handle, (cublasFillMode_t)1, (cublasOperation_t)trans0, (cublasDiagType_t)1,  n, k, x_z, lda, result_z, incx);
  cublasZtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_z, lda, result_z, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incx);
  status = cublasCtpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, result_c, incx);
  cublasCtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incx);
  status = cublasZtpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, result_z, incx);
  cublasZtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incx);
  status = cublasCtpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, result_c, incx);
  cublasCtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incx);
  status = cublasZtpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, result_z, incx);
  cublasZtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trmv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trmv(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incx);
  status = cublasCtrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, lda, result_c, incx);
  cublasCtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trmv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trmv(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incx);
  status = cublasZtrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, lda, result_z, incx);
  cublasZtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsv(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incx);
  status = cublasCtrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, lda, result_c, incx);
  cublasCtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incx));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsv(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incx);
  status = cublasZtrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, lda, result_z, incx);
  cublasZtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hemv(handle->get_queue(), oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy);
  status = cublasChemv(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasChemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hemv(handle->get_queue(), oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy);
  status = cublasZhemv(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZhemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hbmv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hbmv(handle->get_queue(), oneapi::mkl::uplo::lower, n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy);
  status = cublasChbmv(handle, (cublasFillMode_t)fill0, n, k, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasChbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hbmv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hbmv(handle->get_queue(), oneapi::mkl::uplo::lower, n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy);
  status = cublasZhbmv(handle, (cublasFillMode_t)fill0, n, k, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZhbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpmv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpmv(handle->get_queue(), oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy);
  status = cublasChpmv(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, x_c, incx, beta_c, y_c, incy);
  cublasChpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, x_c, incx, beta_c, y_c, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpmv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpmv(handle->get_queue(), oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy);
  status = cublasZhpmv(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, x_z, incx, beta_z, y_z, incy);
  cublasZhpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, x_z, incx, beta_z, y_z, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_f, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her(handle->get_queue(), oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_f, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), lda);
  status = cublasCher(handle, (cublasFillMode_t)fill0, n, alpha_f, x_c, incx, result_c, lda);
  cublasCher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c, lda);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her(handle->get_queue(), oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), lda);
  status = cublasZher(handle, (cublasFillMode_t)fill0, n, alpha_d, x_z, incx, result_z, lda);
  cublasZher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z, lda);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her2(handle->get_queue(), oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), lda);
  status = cublasCher2(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her2(handle->get_queue(), oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), lda);
  status = cublasZher2(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpr(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_f, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c))));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpr(handle->get_queue(), oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_f, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)));
  status = cublasChpr(handle, (cublasFillMode_t)fill0, n, alpha_f, x_c, incx, result_c);
  cublasChpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpr(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z))));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpr(handle->get_queue(), oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)));
  status = cublasZhpr(handle, (cublasFillMode_t)fill0, n, alpha_d, x_z, incx, result_z);
  cublasZhpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpr2(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c))));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpr2(handle->get_queue(), oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)));
  status = cublasChpr2(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, y_c, incy, result_c);
  cublasChpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hpr2(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z))));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hpr2(handle->get_queue(), oneapi::mkl::uplo::lower, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)));
  status = cublasZhpr2(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, y_z, incy, result_z);
  cublasZhpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symv(handle->get_queue(), oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incx, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), incy);
  status = cublasCsymv(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, lda, y_c, incx, beta_c, result_c, incy);
  cublasCsymv(handle, CUBLAS_FILL_MODE_UPPER, n, alpha_c, x_c, lda, y_c, incx, beta_c, result_c, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symv(handle->get_queue(), oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incx, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), incy);
  status = cublasZsymv(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, lda, y_z, incx, beta_z, result_z, incy);
  cublasZsymv(handle, CUBLAS_FILL_MODE_UPPER, n, alpha_z, x_z, lda, y_z, incx, beta_z, result_z, incy);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr(handle->get_queue(), oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), lda);
  status = cublasCsyr(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, result_c, lda);
  cublasCsyr(handle, CUBLAS_FILL_MODE_UPPER, n, alpha_c, x_c, incx, result_c, lda);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr(handle->get_queue(), oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), lda);
  status = cublasZsyr(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, result_z, lda);
  cublasZsyr(handle, CUBLAS_FILL_MODE_UPPER, n, alpha_z, x_z, incx, result_z, lda);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2(handle->get_queue(), oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), lda);
  status = cublasCsyr2(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2(handle->get_queue(), oneapi::mkl::uplo::upper, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), incy, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), lda);
  status = cublasZsyr2(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);


  // level 3
  int N = 100;
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), dpct::get_transpose(trans0), dpct::get_transpose(trans0), N, N, N, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), N, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), N));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), N, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), N);
  status = cublasCgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, N, N, N, alpha_c, x_c, N, y_c, N, beta_c, result_c, N);
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha_c, x_c, N, y_c, N, beta_c, result_c, N);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), dpct::get_transpose(trans0), dpct::get_transpose(trans0), N, N, N, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), N, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), N));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), N, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), N);
  status = cublasZgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, N, N, N, alpha_z, x_z, N, y_z, N, beta_z, result_z, N);
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha_z, x_z, N, y_z, N, beta_z, result_z, N);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, N, N, N, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), N, 16, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), N, 16, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), N, 16, 10));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm_batch(handle->get_queue(), dpct::get_transpose(trans0), dpct::get_transpose(trans1), N, N, N, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), N, 16, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(y_c)), N, 16, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(result_c)), N, 16, 10);
  status = cublasCgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, alpha_c, x_c, N, 16, y_c, N, 16, beta_c, result_c, N, 16, 10);
  cublasCgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, alpha_c, x_c, N, 16, y_c, N, 16, beta_c, result_c, N, 16, 10);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), N, 16, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), N, 16, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), N, 16, 10));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm_batch(handle->get_queue(), dpct::get_transpose(trans0), dpct::get_transpose(trans1), N, N, N, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), N, 16, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(y_z)), N, 16, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(result_z)), N, 16, 10);
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

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), dpct::get_transpose(trans0), dpct::get_transpose(trans0), m, n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc, oneapi::mkl::blas::compute_mode::complex_3m));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, m, n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc, oneapi::mkl::blas::compute_mode::complex_3m);
  status = cublasCgemm3m(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), dpct::get_transpose(trans0), dpct::get_transpose(trans0), m, n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc, oneapi::mkl::blas::compute_mode::complex_3m));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, m, n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc, oneapi::mkl::blas::compute_mode::complex_3m);
  status = cublasZgemm3m(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  int side0 = 0;
  int side1 = 1;
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(handle->get_queue(), (oneapi::mkl::side)side0, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symm(handle->get_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc);
  status = cublasCsymm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(handle->get_queue(), (oneapi::mkl::side)side1, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symm(handle->get_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc);
  status = cublasZsymm(handle, (cublasSideMode_t)side1, (cublasFillMode_t)fill0, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syrk(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc);
  status = cublasCsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);
  cublasCsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syrk(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc);
  status = cublasZsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);
  cublasZsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc);
  status = cublasCsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc);
  status = cublasZsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(handle->get_queue(), oneapi::mkl::side::left, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsm(handle->get_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb);
  status = cublasCtrsm(handle, (cublasSideMode_t)0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, alpha_c, A_c, lda, B_c, ldb);
  cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_c, A_c, lda, B_c, ldb);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(handle->get_queue(), oneapi::mkl::side::right, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsm(handle->get_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb);
  status = cublasZtrsm(handle, (cublasSideMode_t)1, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, alpha_z, A_z, lda, B_z, ldb);
  cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_z, A_z, lda, B_z, ldb);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemm(handle->get_queue(), (oneapi::mkl::side)side0, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hemm(handle->get_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc);
  status = cublasChemm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasChemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::hemm(handle->get_queue(), (oneapi::mkl::side)side0, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::hemm(handle->get_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc);
  status = cublasZhemm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZhemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::herk(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), n, k, alpha_s, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, beta_s, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::herk(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, alpha_s, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, beta_s, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc);
  status = cublasCherk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_s, A_c, lda, &beta_s, C_c, ldc);
  cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &alpha_s, A_c, lda, &beta_s, C_c, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::herk(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), n, k, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, beta_d, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::herk(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_d, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, beta_d, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc);
  status = cublasZherk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_d, A_z, lda, &beta_d, C_z, ldc);
  cublasZherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_d, A_z, lda, &beta_d, C_z, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2k(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, beta_s, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her2k(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, beta_s, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc);
  status = cublasCher2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, B_c, ldb, &beta_s, C_c, ldc);
  cublasCher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, &beta_s, C_c, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::her2k(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, beta_d, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::her2k(handle->get_queue(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, beta_d, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc);
  status = cublasZher2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, B_z, ldb, &beta_d, C_z, ldc);
  cublasZher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, &beta_d, C_z, ldc);

  // CHECK: oneapi::mkl::blas::column_major::trsm(handle->get_queue(), (oneapi::mkl::side)foo(), foo() == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(foo()), (oneapi::mkl::diag)foo(), m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb);
  cublasZtrsm(handle, (cublasSideMode_t)foo(), (cublasFillMode_t)foo(), (cublasOperation_t)foo(), (cublasDiagType_t)foo(), m, n, alpha_z, A_z, lda, B_z, ldb);

  // CHECK: oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::trans, m, n, dpct::get_value(alpha_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(A_c)), lda, dpct::get_value(beta_c, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(B_c)), ldb, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(C_c)), ldc);
  cublasCgeam(handle, CUBLAS_OP_C, CUBLAS_OP_T, m, n, alpha_c, A_c, lda, beta_c, B_c, ldb, C_c, ldc);

  // CHECK: oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::trans, m, n, dpct::get_value(alpha_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(A_z)), lda, dpct::get_value(beta_z, handle->get_queue()), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(B_z)), ldb, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(C_z)), ldc);
  cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_T, m, n, alpha_z, A_z, lda, beta_z, B_z, ldb, C_z, ldc);
}

