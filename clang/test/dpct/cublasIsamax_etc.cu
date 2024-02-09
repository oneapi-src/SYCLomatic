// RUN: dpct --format-range=none --usm-level=none -out-root %T/cublasIsamax_etc %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasIsamax_etc/cublasIsamax_etc.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int foo();

int main() {
  cublasStatus_t status;
  cublasHandle_t handle;
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
  //level1
  //cublasI<t>amax
  // CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::result_memory_t<std::int64_t, int> res(*handle, result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(*handle, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_S)), incx, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: [&]() {
  // CHECK-NEXT: dpct::blas::result_memory_t<std::int64_t, int> res(*handle, result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(*handle, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_S)), incx, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);

  // CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::result_memory_t<std::int64_t, int> res(*handle, result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(*handle, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_D)), incx, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: [&]() {
  // CHECK-NEXT: dpct::blas::result_memory_t<std::int64_t, int> res(*handle, result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(*handle, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_D)), incx, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasIdamax(handle, n, x_D, incx, result);
  cublasIdamax(handle, n, x_D, incx, result);

  //cublasI<t>amin
  // CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::result_memory_t<std::int64_t, int> res(*handle, result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(*handle, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_S)), incx, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: [&]() {
  // CHECK-NEXT: dpct::blas::result_memory_t<std::int64_t, int> res(*handle, result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(*handle, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_S)), incx, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasIsamin(handle, n, x_S, incx, result);
  cublasIsamin(handle, n, x_S, incx, result);

  // CHECK: status = [&]() {
  // CHECK-NEXT: dpct::blas::result_memory_t<std::int64_t, int> res(*handle, result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(*handle, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_D)), incx, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  // CHECK-NEXT: [&]() {
  // CHECK-NEXT: dpct::blas::result_memory_t<std::int64_t, int> res(*handle, result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamin(*handle, n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_D)), incx, res.get_memory(), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  status = cublasIdamin(handle, n, x_D, incx, result);
  cublasIdamin(handle, n, x_D, incx, result);

  //cublas<t>asum
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_S)) {
  // CHECK-NEXT:   result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_S_buf_ct{{[0-9]+}} = sycl::buffer<float>(result_S, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::asum(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_S)) {
  // CHECK-NEXT:   result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_S_buf_ct{{[0-9]+}} = sycl::buffer<float>(result_S, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSasum(handle, n, x_S, incx, result_S);
  cublasSasum(handle, n, x_S, incx, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_D)) {
  // CHECK-NEXT:   result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_D_buf_ct{{[0-9]+}} = sycl::buffer<double>(result_D, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::asum(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_D)) {
  // CHECK-NEXT:   result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_D_buf_ct{{[0-9]+}} = sycl::buffer<double>(result_D, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDasum(handle, n, x_D, incx, result_D);
  cublasDasum(handle, n, x_D, incx, result_D);

  //cublas<t>axpy
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(*handle, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::axpy(*handle, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSaxpy(handle, n, &alpha_S, x_S, incx, result_S, incy);
  cublasSaxpy(handle, n, &alpha_S, x_S, incx, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(*handle, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::axpy(*handle, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDaxpy(handle, n, &alpha_D, x_D, incx, result_D, incy);
  cublasDaxpy(handle, n, &alpha_D, x_D, incx, result_D, incy);

  //cublas<t>copy
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::copy(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::copy(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasScopy(handle, n, x_S, incx, result_S, incy);
  cublasScopy(handle, n, x_S, incx, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::copy(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::copy(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDcopy(handle, n, x_D, incx, result_D, incy);
  cublasDcopy(handle, n, x_D, incx, result_D, incy);

  //cublas<t>dot
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_S)) {
  // CHECK-NEXT:   result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_S_buf_ct{{[0-9]+}} =  sycl::buffer<float>(result_S, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dot(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_S)) {
  // CHECK-NEXT:   result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_S_buf_ct{{[0-9]+}} =  sycl::buffer<float>(result_S, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dot(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSdot(handle, n, x_S, incx, y_S, incy, result_S);
  cublasSdot(handle, n, x_S, incx, y_S, incy, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_D)) {
  // CHECK-NEXT:   result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_D_buf_ct{{[0-9]+}} =  sycl::buffer<double>(result_D, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dot(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_D)) {
  // CHECK-NEXT:   result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_D_buf_ct{{[0-9]+}} =  sycl::buffer<double>(result_D, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::dot(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDdot(handle, n, x_D, incx, y_D, incy, result_D);
  cublasDdot(handle, n, x_D, incx, y_D, incy, result_D);

  //cublas<t>nrm2
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_S)) {
  // CHECK-NEXT:   result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_S_buf_ct{{[0-9]+}} = sycl::buffer<float>(result_S, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::nrm2(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_S)) {
  // CHECK-NEXT:   result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_S_buf_ct{{[0-9]+}} = sycl::buffer<float>(result_S, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSnrm2(handle, n, x_S, incx, result_S);
  cublasSnrm2(handle, n, x_S, incx, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_D)) {
  // CHECK-NEXT:   result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_D_buf_ct{{[0-9]+}} = sycl::buffer<double>(result_D, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::nrm2(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(result_D)) {
  // CHECK-NEXT:   result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   result_D_buf_ct{{[0-9]+}} = sycl::buffer<double>(result_D, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDnrm2(handle, n, x_D, incx, result_D);
  cublasDnrm2(handle, n, x_D, incx, result_D);

  float *x_f = 0;
  float *y_f = 0;
  double *x_d = 0;
  double *y_d = 0;
  //cublas<t>rot
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(*handle, n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy, dpct::get_value(x_S, *handle), dpct::get_value(y_S, *handle)));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rot(*handle, n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy, dpct::get_value(x_S, *handle), dpct::get_value(y_S, *handle));
  // CHECK-NEXT: }
  status = cublasSrot(handle, n, x_f, incx, y_f, incy, x_S, y_S);
  cublasSrot(handle, n, x_f, incx, y_f, incy, x_S, y_S);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rot(*handle, n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy, dpct::get_value(x_D, *handle), dpct::get_value(y_D, *handle)));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rot(*handle, n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy, dpct::get_value(x_D, *handle), dpct::get_value(y_D, *handle));
  // CHECK-NEXT: }
  status = cublasDrot(handle, n, x_d, incx, y_d, incy, x_D, y_D);
  cublasDrot(handle, n, x_d, incx, y_d, incy, x_D, y_D);

  //cublas<t>rotg
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(x_f)) {
  // CHECK-NEXT:   x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT:   x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   x_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(x_f, sycl::range<1>(1));
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(y_f, sycl::range<1>(1));
  // CHECK-NEXT:   x_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(x_f, sycl::range<1>(1));
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(y_f, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rotg(*handle, x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(x_f)) {
  // CHECK-NEXT:   x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT:   x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   x_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(x_f, sycl::range<1>(1));
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(y_f, sycl::range<1>(1));
  // CHECK-NEXT:   x_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(x_f, sycl::range<1>(1));
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(y_f, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotg(*handle, x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSrotg(handle, x_f, y_f, x_f, y_f);
  cublasSrotg(handle, x_f, y_f, x_f, y_f);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(x_d)) {
  // CHECK-NEXT:   x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT:   x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   x_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(x_d, sycl::range<1>(1));
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(y_d, sycl::range<1>(1));
  // CHECK-NEXT:   x_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(x_d, sycl::range<1>(1));
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(y_d, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rotg(*handle, x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(x_d)) {
  // CHECK-NEXT:   x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT:   x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   x_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(x_d, sycl::range<1>(1));
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(y_d, sycl::range<1>(1));
  // CHECK-NEXT:   x_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(x_d, sycl::range<1>(1));
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(y_d, sycl::range<1>(1));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotg(*handle, x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDrotg(handle, x_d, y_d, x_d, y_d);
  cublasDrotg(handle, x_d, y_d, x_d, y_d);

  //cublas<t>rotm
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rotm(*handle, n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy, x_S_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotm(*handle, n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy, x_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSrotm(handle, n, x_f, incx, y_f, incy, x_S);
  cublasSrotm(handle, n, x_f, incx, y_f, incy, x_S);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rotm(*handle, n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy, x_D_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotm(*handle, n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy, x_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDrotm(handle, n, x_d, incx, y_d, incy, x_D);
  cublasDrotm(handle, n, x_d, incx, y_d, incy, x_D);

  //cublas<t>rotmg
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(x_f)) {
  // CHECK-NEXT:   x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   x_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(x_f, sycl::range<1>(1));
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(y_f, sycl::range<1>(1));
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(y_f, sycl::range<1>(1));
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(y_f, sycl::range<1>(5));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rotmg(*handle, x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, dpct::get_value(x_S, *handle), y_f_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(x_f)) {
  // CHECK-NEXT:   x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   x_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(x_f, sycl::range<1>(1));
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(y_f, sycl::range<1>(1));
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(y_f, sycl::range<1>(1));
  // CHECK-NEXT:   y_f_buf_ct{{[0-9]+}} = sycl::buffer<float>(y_f, sycl::range<1>(5));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotmg(*handle, x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, dpct::get_value(x_S, *handle), y_f_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSrotmg(handle, x_f, y_f, y_f, x_S, y_f);
  cublasSrotmg(handle, x_f, y_f, y_f, x_S, y_f);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(x_d)) {
  // CHECK-NEXT:   x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   x_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(x_d, sycl::range<1>(1));
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(y_d, sycl::range<1>(1));
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(y_d, sycl::range<1>(1));
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(y_d, sycl::range<1>(5));
  // CHECK-NEXT: }
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rotmg(*handle, x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, dpct::get_value(x_D, *handle), y_d_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(sycl::range<1>(1));
  // CHECK-NEXT: if (dpct::is_device_ptr(x_d)) {
  // CHECK-NEXT:   x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   x_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(x_d, sycl::range<1>(1));
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(y_d, sycl::range<1>(1));
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(y_d, sycl::range<1>(1));
  // CHECK-NEXT:   y_d_buf_ct{{[0-9]+}} = sycl::buffer<double>(y_d, sycl::range<1>(5));
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotmg(*handle, x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, dpct::get_value(x_D, *handle), y_d_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDrotmg(handle, x_d, y_d, y_d, x_D, y_d);
  cublasDrotmg(handle, x_d, y_d, y_d, x_D, y_d);

  //cublas<t>scal
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(*handle, n, alpha_S, x_f_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(*handle, n, alpha_S, x_f_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasSscal(handle, n, &alpha_S, x_f, incx);
  cublasSscal(handle, n, &alpha_S, x_f, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(*handle, n, alpha_D, x_d_buf_ct{{[0-9]+}}, incx));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::scal(*handle, n, alpha_D, x_d_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasDscal(handle, n, &alpha_D, x_d, incx);
  cublasDscal(handle, n, &alpha_D, x_d, incx);

  //cublas<t>swap
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::swap(*handle, n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::swap(*handle, n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSswap(handle, n, x_f, incx, y_f, incy);
  cublasSswap(handle, n, x_f, incx, y_f, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::swap(*handle, n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::swap(*handle, n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDswap(handle, n, x_d, incx, y_d, incy);
  cublasDswap(handle, n, x_d, incx, y_d, incy);

  int trans0 = 0;
  int trans1 = 1;
  int trans2 = 2;
  //level2
  //cublas<t>gbmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gbmv(*handle, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, m, n, m, n, alpha_S, x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gbmv(*handle, oneapi::mkl::transpose::nontrans, m, n, m, n, alpha_S, x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSgbmv(handle, (cublasOperation_t)trans0, m, n, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gbmv(*handle, trans1==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans1, m, n, m, n, alpha_D, x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gbmv(*handle, oneapi::mkl::transpose::nontrans, m, n, m, n, alpha_D, x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDgbmv(handle, (cublasOperation_t)trans1, m, n, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>gemv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemv(*handle, trans2==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans2, m, n, alpha_S, x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemv(*handle, oneapi::mkl::transpose::nontrans, m, n, alpha_S, x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSgemv(handle, (cublasOperation_t)trans2, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemv(*handle, oneapi::mkl::transpose::nontrans, m, n, alpha_D, x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemv(*handle, oneapi::mkl::transpose::nontrans, m, n, alpha_D, x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDgemv(handle, (cublasOperation_t)0, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>ger
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::ger(*handle, m, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::ger(*handle, m, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasSger(handle, m, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  cublasSger(handle, m, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::ger(*handle, m, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::ger(*handle, m, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasDger(handle, m, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  cublasDger(handle, m, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);

  int fill0 = 0;
  int fill1 = 1;
  //cublas<t>sbmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::sbmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, m, n, alpha_S, x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::sbmv(*handle, oneapi::mkl::uplo::upper, m, n, alpha_S, x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSsbmv(handle, (cublasFillMode_t)fill0, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::sbmv(*handle, fill1==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, m, n, alpha_D, x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::sbmv(*handle, oneapi::mkl::uplo::upper, m, n, alpha_D, x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDsbmv(handle, (cublasFillMode_t)fill1, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>spmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::spmv(*handle, oneapi::mkl::uplo::lower, n, alpha_S, x_S_buf_ct{{[0-9]+}}, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::spmv(*handle, oneapi::mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSspmv(handle, (cublasFillMode_t)0, n, &alpha_S, x_S, y_S, incx, &beta_S, result_S, incy);
  cublasSspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, y_S, incx, &beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::spmv(*handle, oneapi::mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::spmv(*handle, oneapi::mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDspmv(handle, (cublasFillMode_t)1, n, &alpha_D, x_D, y_D, incx, &beta_D, result_D, incy);
  cublasDspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>spr
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::spr(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::spr(*handle, oneapi::mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSspr(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, incx, result_S);
  cublasSspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::spr(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::spr(*handle, oneapi::mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDspr(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, incx, result_D);
  cublasDspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D);

  //cublas<t>spr2
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::spr2(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::spr2(*handle, oneapi::mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSspr2(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, incx, y_S, incy, result_S);
  cublasSspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::spr2(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::spr2(*handle, oneapi::mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDspr2(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, incx, y_D, incy, result_D);
  cublasDspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D);

  //cublas<t>symv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symv(*handle, oneapi::mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, beta_S, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSsymv(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symv(*handle, oneapi::mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, beta_D, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDsymv(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>syr
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr(*handle, oneapi::mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasSsyr(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, incx, result_S, lda);
  cublasSsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr(*handle, oneapi::mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasDsyr(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, incx, result_D, lda);
  cublasDsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D, lda);

  //cublas<t>syr2
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2(*handle, oneapi::mkl::uplo::upper, n, alpha_S, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasSsyr2(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  cublasSsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}, lda));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2(*handle, oneapi::mkl::uplo::upper, n, alpha_D, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasDsyr2(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  cublasDsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);

  int diag0 = 0;
  int diag1 = 1;
  //cublas<t>tbmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, (oneapi::mkl::diag)diag0, n, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbmv(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStbmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)1, (cublasDiagType_t)diag0, n, n, x_S, lda, result_S, incy);
  cublasStbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, (oneapi::mkl::diag)diag1, n, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbmv(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtbmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)2, (cublasDiagType_t)diag1, n, n, x_D, lda, result_D, incy);
  cublasDtbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);

  //cublas<t>tbsv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbsv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, oneapi::mkl::diag::nonunit, n, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbsv(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStbsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)0, n, n, x_S, lda, result_S, incy);
  cublasStbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tbsv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, oneapi::mkl::diag::unit, n, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tbsv(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtbsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)1, n, n, x_D, lda, result_D, incy);
  cublasDtbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);

  //cublas<t>tpmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_S_buf_ct{{[0-9]+}}, result_S_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpmv(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_S_buf_ct{{[0-9]+}}, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_S, result_S, incy);
  cublasStpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_D_buf_ct{{[0-9]+}}, result_D_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpmv(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_D_buf_ct{{[0-9]+}}, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_D, result_D, incy);
  cublasDtpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);

  //cublas<t>tpsv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpsv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_S_buf_ct{{[0-9]+}}, result_S_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpsv(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_S_buf_ct{{[0-9]+}}, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_S, result_S, incy);
  cublasStpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::tpsv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_D_buf_ct{{[0-9]+}}, result_D_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::tpsv(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_D_buf_ct{{[0-9]+}}, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_D, result_D, incy);
  cublasDtpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);

  //cublas<t>trmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trmv(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_S, lda, result_S, incy);
  cublasStrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trmv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trmv(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_D, lda, result_D, incy);
  cublasDtrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, lda, result_D, incy);

  //cublas<t>trsv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsv(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_S, lda, result_S, incy);
  cublasStrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsv(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsv(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_D, lda, result_D, incy);
  cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, lda, result_D, incy);

  //level3
  int side0 = 0;
  int side1 = 1;
  // cublas<T>gemmStridedBatched
  // CHECK: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(B_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, n, n, n, alpha_S, A_S_buf_ct{{[0-9]+}}, n, 16, B_S_buf_ct{{[0-9]+}}, n, 16, beta_S, C_S_buf_ct{{[0-9]+}}, n, 16, 10));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(B_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm_batch(*handle, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, trans1==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans1, n, n, n, alpha_S, A_S_buf_ct{{[0-9]+}}, n, 16, B_S_buf_ct{{[0-9]+}}, n, 16, beta_S, C_S_buf_ct{{[0-9]+}}, n, 16, 10);
  // CHECK-NEXT: }
  status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha_S, A_S, n, 16, B_S, n, 16, &beta_S, C_S, n, 16, 10);
  cublasSgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, n, n, n, &alpha_S, A_S, n, 16, B_S, n, 16, &beta_S, C_S, n, 16, 10);

  // CHECK: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto B_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(*handle, oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, n, n, n, alpha_D, A_D_buf_ct{{[0-9]+}}, n, 16, B_D_buf_ct{{[0-9]+}}, n, 16, beta_D, C_D_buf_ct{{[0-9]+}}, n, 16, 10));
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto B_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm_batch(*handle, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, trans1==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans1, n, n, n, alpha_D, A_D_buf_ct{{[0-9]+}}, n, 16, B_D_buf_ct{{[0-9]+}}, n, 16, beta_D, C_D_buf_ct{{[0-9]+}}, n, 16, 10);
  // CHECK-NEXT: }
  status = cublasDgemmStridedBatched(handle, CUBLAS_OP_C, CUBLAS_OP_C, n, n, n, &alpha_D, A_D, n, 16, B_D, n, 16, &beta_D, C_D, n, 16, 10);
  cublasDgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, n, n, n, &alpha_D, A_D, n, 16, B_D, n, 16, &beta_D, C_D, n, 16, 10);

  __half alpha_H, beta_H;
  __half* A_H, *B_H, *C_H;
  // CHECK: {
  // CHECK-NEXT: auto A_H_buf_ct{{[0-9]+}} = dpct::get_buffer<sycl::half>(A_H);
  // CHECK-NEXT: auto B_H_buf_ct{{[0-9]+}} = dpct::get_buffer<sycl::half>(B_H);
  // CHECK-NEXT: auto C_H_buf_ct{{[0-9]+}} = dpct::get_buffer<sycl::half>(C_H);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm_batch(*handle, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, trans1==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans1, n, n, n, alpha_H, A_H_buf_ct{{[0-9]+}}, n, 16, B_H_buf_ct{{[0-9]+}}, n, 16, beta_H, C_H_buf_ct{{[0-9]+}}, n, 16, 10);
  // CHECK-NEXT: }
  cublasHgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, n, n, n, &alpha_H, A_H, n, 16, B_H, n, 16, &beta_H, C_H, n, 16, 10);

  const float** A_S_array;
  const float** B_S_array;
  float** C_S_array;
  const double** A_D_array;
  const double** B_D_array;
  double** C_D_array;
  const __half** A_H_array;
  const __half** B_H_array;
  __half** C_H_array;
  cublasOperation_t trans3 = CUBLAS_OP_N;

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cublasSgemmBatched is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = cublasSgemmBatched(handle, trans3, trans3, n, n, n, &alpha_S, A_S_array, n, B_S_array, n, &beta_S, C_S_array, n, 10);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cublasSgemmBatched is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasSgemmBatched(handle, trans3, trans3, n, n, n, &alpha_S, A_S_array, n, B_S_array, n, &beta_S, C_S_array, n, 10);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cublasDgemmBatched is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = cublasDgemmBatched(handle, trans3, trans3, n, n, n, &alpha_D, A_D_array, n, B_D_array, n, &beta_D, C_D_array, n, 10);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cublasDgemmBatched is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasDgemmBatched(handle, trans3, trans3, n, n, n, &alpha_D, A_D_array, n, B_D_array, n, &beta_D, C_D_array, n, 10);
  status = cublasSgemmBatched(handle, trans3, trans3, n, n, n, &alpha_S, A_S_array, n, B_S_array, n, &beta_S, C_S_array, n, 10);
  cublasSgemmBatched(handle, trans3, trans3, n, n, n, &alpha_S, A_S_array, n, B_S_array, n, &beta_S, C_S_array, n, 10);
  status = cublasDgemmBatched(handle, trans3, trans3, n, n, n, &alpha_D, A_D_array, n, B_D_array, n, &beta_D, C_D_array, n, 10);
  cublasDgemmBatched(handle, trans3, trans3, n, n, n, &alpha_D, A_D_array, n, B_D_array, n, &beta_D, C_D_array, n, 10);

  // cublas<T>symm
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(*handle, (oneapi::mkl::side)side0, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, m, n, dpct::get_value(&alpha_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_S)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(B_S)), ldb, dpct::get_value(&beta_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_S)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symm(*handle, oneapi::mkl::side::right, oneapi::mkl::uplo::lower, m, n, dpct::get_value(&alpha_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_S)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(B_S)), ldb, dpct::get_value(&beta_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_S)), ldc);
  status = cublasSsymm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);
  cublasSsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, m, n, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symm(*handle, (oneapi::mkl::side)side1, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, m, n, dpct::get_value(&alpha_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_D)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(B_D)), ldb, dpct::get_value(&beta_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_D)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symm(*handle, oneapi::mkl::side::right, oneapi::mkl::uplo::lower, m, n, dpct::get_value(&alpha_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_D)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(B_D)), ldb, dpct::get_value(&beta_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_D)), ldc);
  status = cublasDsymm(handle, (cublasSideMode_t)side1, (cublasFillMode_t)fill0, m, n, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);
  cublasDsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, m, n, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);

  // cublas<T>syrk
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(*handle, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), n, k, dpct::get_value(&alpha_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_S)), lda, dpct::get_value(&beta_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_S)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syrk(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_S)), lda, dpct::get_value(&beta_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_S)), ldc);
  status = cublasSsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_S, A_S, lda, &beta_S, C_S, ldc);
  cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, &beta_S, C_S, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syrk(*handle, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), n, k, dpct::get_value(&alpha_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_D)), lda, dpct::get_value(&beta_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_D)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syrk(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_D)), lda, dpct::get_value(&beta_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_D)), ldc);
  status = cublasDsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_D, A_D, lda, &beta_D, C_D, ldc);
  cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, &beta_D, C_D, ldc);

  // cublas<T>syr2k
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(*handle, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), n, k, dpct::get_value(&alpha_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_S)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(B_S)), ldb, dpct::get_value(&beta_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_S)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2k(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_S)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(B_S)), ldb, dpct::get_value(&beta_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_S)), ldc);
  status = cublasSsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);
  cublasSsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2k(*handle, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), n, k, dpct::get_value(&alpha_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_D)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(B_D)), ldb, dpct::get_value(&beta_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_D)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2k(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_D)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(B_D)), ldb, dpct::get_value(&beta_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_D)), ldc);
  status = cublasDsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);
  cublasDsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);

  // cublas<T>trsm
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(*handle, oneapi::mkl::side::left, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, m, n, dpct::get_value(&alpha_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_S)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_S)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, dpct::get_value(&alpha_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_S)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_S)), ldc);
  status = cublasStrsm(handle, (cublasSideMode_t)0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, &alpha_S, A_S, lda, C_S, ldc);
  cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_S, A_S, lda, C_S, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::trsm(*handle, oneapi::mkl::side::right, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, m, n, dpct::get_value(&alpha_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_D)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_D)), ldc));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::trsm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, dpct::get_value(&alpha_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_D)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_D)), ldc);
  status = cublasDtrsm(handle, (cublasSideMode_t)1, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, &alpha_D, A_D, lda, C_D, ldc);
  cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_D, A_D, lda, C_D, ldc);


  // CHECK:   oneapi::mkl::blas::column_major::trsm(*handle, (oneapi::mkl::side)foo(), foo() == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(foo()), (oneapi::mkl::diag)foo(), m, n, dpct::get_value(&alpha_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_D)), lda, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_D)), ldc);
  cublasDtrsm(handle, (cublasSideMode_t)foo(), (cublasFillMode_t)foo(), (cublasOperation_t)foo(), (cublasDiagType_t)foo(), m, n, &alpha_D, A_D, lda, C_D, ldc);

  // CHECK: oneapi::mkl::blas::column_major::omatadd(*handle, oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::trans, m, n, dpct::get_value(&alpha_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(A_S)), lda, dpct::get_value(&beta_S, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(B_S)), ldb, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(C_S)), ldc);
  cublasSgeam(handle, CUBLAS_OP_C, CUBLAS_OP_T, m, n, &alpha_S, A_S, lda, &beta_S, B_S, ldb, C_S, ldc);

  // CHECK: oneapi::mkl::blas::column_major::omatadd(*handle, oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::trans, m, n, dpct::get_value(&alpha_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(A_D)), lda, dpct::get_value(&beta_D, *handle), dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(B_D)), ldb, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(C_D)), ldc);
  cublasDgeam(handle, CUBLAS_OP_C, CUBLAS_OP_T, m, n, &alpha_D, A_D, lda, &beta_D, B_D, ldb, C_D, ldc);
  return 0;
}

