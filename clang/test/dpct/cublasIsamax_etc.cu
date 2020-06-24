// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasIsamax_etc.dp.cpp --match-full-lines %s
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
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::iamax(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamax(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::iamax(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamax(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIdamax(handle, n, x_D, incx, result);
  cublasIdamax(handle, n, x_D, incx, result);

  //cublasI<t>amin
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::iamin(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamin(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIsamin(handle, n, x_S, incx, result);
  cublasIsamin(handle, n, x_S, incx, result);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::iamin(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
  // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamin(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, res_temp_buf_ct{{[0-9]+}});
  // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIdamin(handle, n, x_D, incx, result);
  cublasIdamin(handle, n, x_D, incx, result);

  //cublas<t>asum
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::asum(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::asum(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSasum(handle, n, x_S, incx, result_S);
  cublasSasum(handle, n, x_S, incx, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::asum(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::asum(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDasum(handle, n, x_D, incx, result_D);
  cublasDasum(handle, n, x_D, incx, result_D);

  //cublas<t>axpy
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::axpy(*handle, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::axpy(*handle, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSaxpy(handle, n, &alpha_S, x_S, incx, result_S, incy);
  cublasSaxpy(handle, n, &alpha_S, x_S, incx, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::axpy(*handle, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::axpy(*handle, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDaxpy(handle, n, &alpha_D, x_D, incx, result_D, incy);
  cublasDaxpy(handle, n, &alpha_D, x_D, incx, result_D, incy);

  //cublas<t>copy
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::copy(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::copy(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasScopy(handle, n, x_S, incx, result_S, incy);
  cublasScopy(handle, n, x_S, incx, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::copy(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::copy(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDcopy(handle, n, x_D, incx, result_D, incy);
  cublasDcopy(handle, n, x_D, incx, result_D, incy);

  //cublas<t>dot
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::dot(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::dot(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSdot(handle, n, x_S, incx, y_S, incy, result_S);
  cublasSdot(handle, n, x_S, incx, y_S, incy, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::dot(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::dot(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDdot(handle, n, x_D, incx, y_D, incy, result_D);
  cublasDdot(handle, n, x_D, incx, y_D, incy, result_D);

  //cublas<t>nrm2
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::nrm2(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::nrm2(*handle, n, x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSnrm2(handle, n, x_S, incx, result_S);
  cublasSnrm2(handle, n, x_S, incx, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::nrm2(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::nrm2(*handle, n, x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}});
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
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::rot(*handle, n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy, dpct::get_value(x_S, *handle), dpct::get_value(y_S, *handle)), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: mkl::blas::rot(*handle, n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy, dpct::get_value(x_S, *handle), dpct::get_value(y_S, *handle));
  // CHECK-NEXT: }
  status = cublasSrot(handle, n, x_f, incx, y_f, incy, x_S, y_S);
  cublasSrot(handle, n, x_f, incx, y_f, incy, x_S, y_S);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::rot(*handle, n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy, dpct::get_value(x_D, *handle), dpct::get_value(y_D, *handle)), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: mkl::blas::rot(*handle, n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy, dpct::get_value(x_D, *handle), dpct::get_value(y_D, *handle));
  // CHECK-NEXT: }
  status = cublasDrot(handle, n, x_d, incx, y_d, incy, x_D, y_D);
  cublasDrot(handle, n, x_d, incx, y_d, incy, x_D, y_D);

  //cublas<t>rotg
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::rotg(*handle, x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: mkl::blas::rotg(*handle, x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSrotg(handle, x_f, y_f, x_f, y_f);
  cublasSrotg(handle, x_f, y_f, x_f, y_f);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::rotg(*handle, x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: mkl::blas::rotg(*handle, x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDrotg(handle, x_d, y_d, x_d, y_d);
  cublasDrotg(handle, x_d, y_d, x_d, y_d);

  //cublas<t>rotm
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::rotm(*handle, n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy, x_S_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: mkl::blas::rotm(*handle, n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy, x_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSrotm(handle, n, x_f, incx, y_f, incy, x_S);
  cublasSrotm(handle, n, x_f, incx, y_f, incy, x_S);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::rotm(*handle, n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy, x_D_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: mkl::blas::rotm(*handle, n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy, x_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDrotm(handle, n, x_d, incx, y_d, incy, x_D);
  cublasDrotm(handle, n, x_d, incx, y_d, incy, x_D);

  //cublas<t>rotmg
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::rotmg(*handle, x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, dpct::get_value(x_S, *handle), y_f_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: mkl::blas::rotmg(*handle, x_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, y_f_buf_ct{{[0-9]+}}, dpct::get_value(x_S, *handle), y_f_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSrotmg(handle, x_f, y_f, y_f, x_S, y_f);
  cublasSrotmg(handle, x_f, y_f, y_f, x_S, y_f);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::rotmg(*handle, x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, dpct::get_value(x_D, *handle), y_d_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: mkl::blas::rotmg(*handle, x_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, y_d_buf_ct{{[0-9]+}}, dpct::get_value(x_D, *handle), y_d_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDrotmg(handle, x_d, y_d, y_d, x_D, y_d);
  cublasDrotmg(handle, x_d, y_d, y_d, x_D, y_d);

  //cublas<t>scal
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::scal(*handle, n, dpct::get_value(&alpha_S, *handle), x_f_buf_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: mkl::blas::scal(*handle, n, dpct::get_value(&alpha_S, *handle), x_f_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasSscal(handle, n, &alpha_S, x_f, incx);
  cublasSscal(handle, n, &alpha_S, x_f, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::scal(*handle, n, dpct::get_value(&alpha_D, *handle), x_d_buf_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: mkl::blas::scal(*handle, n, dpct::get_value(&alpha_D, *handle), x_d_buf_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasDscal(handle, n, &alpha_D, x_d, incx);
  cublasDscal(handle, n, &alpha_D, x_d, incx);

  //cublas<t>swap
  // CHECK: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::swap(*handle, n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_f);
  // CHECK-NEXT: auto y_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_f);
  // CHECK-NEXT: mkl::blas::swap(*handle, n, x_f_buf_ct{{[0-9]+}}, incx, y_f_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSswap(handle, n, x_f, incx, y_f, incy);
  cublasSswap(handle, n, x_f, incx, y_f, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::swap(*handle, n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_d);
  // CHECK-NEXT: auto y_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_d);
  // CHECK-NEXT: mkl::blas::swap(*handle, n, x_d_buf_ct{{[0-9]+}}, incx, y_d_buf_ct{{[0-9]+}}, incy);
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
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gbmv(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, m, n, m, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_S, *handle), result_S_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::gbmv(*handle, mkl::transpose::nontrans, m, n, m, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_S, *handle), result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSgbmv(handle, (cublasOperation_t)trans0, m, n, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gbmv(*handle, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, m, n, m, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_D, *handle), result_D_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::gbmv(*handle, mkl::transpose::nontrans, m, n, m, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_D, *handle), result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDgbmv(handle, (cublasOperation_t)trans1, m, n, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>gemv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gemv(*handle, trans2==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans2, m, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_S, *handle), result_S_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::gemv(*handle, mkl::transpose::nontrans, m, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_S, *handle), result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSgemv(handle, (cublasOperation_t)trans2, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gemv(*handle, mkl::transpose::nontrans, m, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_D, *handle), result_D_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::gemv(*handle, mkl::transpose::nontrans, m, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_D, *handle), result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDgemv(handle, (cublasOperation_t)0, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>ger
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::ger(*handle, m, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::ger(*handle, m, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasSger(handle, m, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  cublasSger(handle, m, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::ger(*handle, m, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::ger(*handle, m, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}, lda);
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
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::sbmv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, m, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_S, *handle), result_S_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::sbmv(*handle, mkl::uplo::upper, m, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_S, *handle), result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSsbmv(handle, (cublasFillMode_t)fill0, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::sbmv(*handle, fill1==0 ? mkl::uplo::lower : mkl::uplo::upper, m, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_D, *handle), result_D_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::sbmv(*handle, mkl::uplo::upper, m, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_D, *handle), result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDsbmv(handle, (cublasFillMode_t)fill1, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>spmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::spmv(*handle, mkl::uplo::lower, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, y_S_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_S, *handle), result_S_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::spmv(*handle, mkl::uplo::upper, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, y_S_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_S, *handle), result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSspmv(handle, (cublasFillMode_t)0, n, &alpha_S, x_S, y_S, incx, &beta_S, result_S, incy);
  cublasSspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, y_S, incx, &beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::spmv(*handle, mkl::uplo::upper, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, y_D_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_D, *handle), result_D_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::spmv(*handle, mkl::uplo::upper, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, y_D_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_D, *handle), result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDspmv(handle, (cublasFillMode_t)1, n, &alpha_D, x_D, y_D, incx, &beta_D, result_D, incy);
  cublasDspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>spr
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::spr(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::spr(*handle, mkl::uplo::upper, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSspr(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, incx, result_S);
  cublasSspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::spr(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::spr(*handle, mkl::uplo::upper, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDspr(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, incx, result_D);
  cublasDspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D);

  //cublas<t>spr2
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::spr2(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::spr2(*handle, mkl::uplo::upper, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasSspr2(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, incx, y_S, incy, result_S);
  cublasSspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::spr2(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::spr2(*handle, mkl::uplo::upper, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasDspr2(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, incx, y_D, incy, result_D);
  cublasDspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D);

  //cublas<t>symv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::symv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_S, *handle), result_S_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::symv(*handle, mkl::uplo::upper, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, lda, y_S_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_S, *handle), result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSsymv(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::symv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_D, *handle), result_D_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::symv(*handle, mkl::uplo::upper, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, lda, y_D_buf_ct{{[0-9]+}}, incx, dpct::get_value(&beta_D, *handle), result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDsymv(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>syr
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::syr(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::syr(*handle, mkl::uplo::upper, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, incx, result_S_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasSsyr(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, incx, result_S, lda);
  cublasSsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::syr(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::syr(*handle, mkl::uplo::upper, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, incx, result_D_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasDsyr(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, incx, result_D, lda);
  cublasDsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D, lda);

  //cublas<t>syr2
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::syr2(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto y_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::syr2(*handle, mkl::uplo::upper, n, dpct::get_value(&alpha_S, *handle), x_S_buf_ct{{[0-9]+}}, incx, y_S_buf_ct{{[0-9]+}}, incy, result_S_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasSsyr2(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  cublasSsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::syr2(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto y_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(y_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::syr2(*handle, mkl::uplo::upper, n, dpct::get_value(&alpha_D, *handle), x_D_buf_ct{{[0-9]+}}, incx, y_D_buf_ct{{[0-9]+}}, incy, result_D_buf_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasDsyr2(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  cublasDsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);

  int diag0 = 0;
  int diag1 = 1;
  //cublas<t>tbmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tbmv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, mkl::transpose::trans, (mkl::diag)diag0, n, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::tbmv(*handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStbmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)1, (cublasDiagType_t)diag0, n, n, x_S, lda, result_S, incy);
  cublasStbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tbmv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, mkl::transpose::conjtrans, (mkl::diag)diag1, n, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::tbmv(*handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtbmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)2, (cublasDiagType_t)diag1, n, n, x_D, lda, result_D, incy);
  cublasDtbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);

  //cublas<t>tbsv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tbsv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, mkl::diag::nonunit, n, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::tbsv(*handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStbsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)0, n, n, x_S, lda, result_S, incy);
  cublasStbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tbsv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, mkl::diag::unit, n, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::tbsv(*handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtbsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)1, n, n, x_D, lda, result_D, incy);
  cublasDtbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);

  //cublas<t>tpmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tpmv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_S_buf_ct{{[0-9]+}}, result_S_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::tpmv(*handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_buf_ct{{[0-9]+}}, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_S, result_S, incy);
  cublasStpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tpmv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_D_buf_ct{{[0-9]+}}, result_D_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::tpmv(*handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_buf_ct{{[0-9]+}}, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_D, result_D, incy);
  cublasDtpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);

  //cublas<t>tpsv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tpsv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_S_buf_ct{{[0-9]+}}, result_S_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::tpsv(*handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_buf_ct{{[0-9]+}}, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_S, result_S, incy);
  cublasStpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::tpsv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_D_buf_ct{{[0-9]+}}, result_D_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::tpsv(*handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_buf_ct{{[0-9]+}}, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_D, result_D, incy);
  cublasDtpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);

  //cublas<t>trmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::trmv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::trmv(*handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_S, lda, result_S, incy);
  cublasStrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::trmv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::trmv(*handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_D, lda, result_D, incy);
  cublasDtrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, lda, result_D, incy);

  //cublas<t>trsv
  // CHECK: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::trsv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
  // CHECK-NEXT: auto result_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(result_S);
  // CHECK-NEXT: mkl::blas::trsv(*handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_buf_ct{{[0-9]+}}, lda, result_S_buf_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_S, lda, result_S, incy);
  cublasStrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::trsv(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(x_D);
  // CHECK-NEXT: auto result_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(result_D);
  // CHECK-NEXT: mkl::blas::trsv(*handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_buf_ct{{[0-9]+}}, lda, result_D_buf_ct{{[0-9]+}}, incy);
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
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gemm_batch(*handle, mkl::transpose::nontrans, mkl::transpose::trans, n, n, n, dpct::get_value(&alpha_S, *handle), A_S_buf_ct{{[0-9]+}}, n, 16, B_S_buf_ct{{[0-9]+}}, n, 16, dpct::get_value(&beta_S, *handle), C_S_buf_ct{{[0-9]+}}, n, 16, 10), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(B_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: mkl::blas::gemm_batch(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, n, n, n, dpct::get_value(&alpha_S, *handle), A_S_buf_ct{{[0-9]+}}, n, 16, B_S_buf_ct{{[0-9]+}}, n, 16, dpct::get_value(&beta_S, *handle), C_S_buf_ct{{[0-9]+}}, n, 16, 10);
  // CHECK-NEXT: }
  status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha_S, A_S, n, 16, B_S, n, 16, &beta_S, C_S, n, 16, 10);
  cublasSgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, n, n, n, &alpha_S, A_S, n, 16, B_S, n, 16, &beta_S, C_S, n, 16, 10);

  // CHECK: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto B_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gemm_batch(*handle, mkl::transpose::conjtrans, mkl::transpose::conjtrans, n, n, n, dpct::get_value(&alpha_D, *handle), A_D_buf_ct{{[0-9]+}}, n, 16, B_D_buf_ct{{[0-9]+}}, n, 16, dpct::get_value(&beta_D, *handle), C_D_buf_ct{{[0-9]+}}, n, 16, 10), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto B_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: mkl::blas::gemm_batch(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, n, n, n, dpct::get_value(&alpha_D, *handle), A_D_buf_ct{{[0-9]+}}, n, 16, B_D_buf_ct{{[0-9]+}}, n, 16, dpct::get_value(&beta_D, *handle), C_D_buf_ct{{[0-9]+}}, n, 16, 10);
  // CHECK-NEXT: }
  status = cublasDgemmStridedBatched(handle, CUBLAS_OP_C, CUBLAS_OP_C, n, n, n, &alpha_D, A_D, n, 16, B_D, n, 16, &beta_D, C_D, n, 16, 10);
  cublasDgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, n, n, n, &alpha_D, A_D, n, 16, B_D, n, 16, &beta_D, C_D, n, 16, 10);

  __half alpha_H, beta_H;
  __half* A_H, *B_H, *C_H;
  // CHECK: {
  // CHECK-NEXT: auto A_H_buf_ct{{[0-9]+}} = dpct::get_buffer<sycl::half>(A_H);
  // CHECK-NEXT: auto B_H_buf_ct{{[0-9]+}} = dpct::get_buffer<sycl::half>(B_H);
  // CHECK-NEXT: auto C_H_buf_ct{{[0-9]+}} = dpct::get_buffer<sycl::half>(C_H);
  // CHECK-NEXT: mkl::blas::gemm_batch(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, n, n, n, dpct::get_value(&alpha_H, *handle), A_H_buf_ct{{[0-9]+}}, n, 16, B_H_buf_ct{{[0-9]+}}, n, 16, dpct::get_value(&beta_H, *handle), C_H_buf_ct{{[0-9]+}}, n, 16, 10);
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
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of this CUDA API is not supported by the Intel(R) DPC++ Compatibility Tool.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = cublasSgemmBatched(handle, trans3, trans3, n, n, n, &alpha_S, A_S_array, n, B_S_array, n, &beta_S, C_S_array, n, 10);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of this CUDA API is not supported by the Intel(R) DPC++ Compatibility Tool.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasSgemmBatched(handle, trans3, trans3, n, n, n, &alpha_S, A_S_array, n, B_S_array, n, &beta_S, C_S_array, n, 10);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of this CUDA API is not supported by the Intel(R) DPC++ Compatibility Tool.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = cublasDgemmBatched(handle, trans3, trans3, n, n, n, &alpha_D, A_D_array, n, B_D_array, n, &beta_D, C_D_array, n, 10);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of this CUDA API is not supported by the Intel(R) DPC++ Compatibility Tool.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasDgemmBatched(handle, trans3, trans3, n, n, n, &alpha_D, A_D_array, n, B_D_array, n, &beta_D, C_D_array, n, 10);
  status = cublasSgemmBatched(handle, trans3, trans3, n, n, n, &alpha_S, A_S_array, n, B_S_array, n, &beta_S, C_S_array, n, 10);
  cublasSgemmBatched(handle, trans3, trans3, n, n, n, &alpha_S, A_S_array, n, B_S_array, n, &beta_S, C_S_array, n, 10);
  status = cublasDgemmBatched(handle, trans3, trans3, n, n, n, &alpha_D, A_D_array, n, B_D_array, n, &beta_D, C_D_array, n, 10);
  cublasDgemmBatched(handle, trans3, trans3, n, n, n, &alpha_D, A_D_array, n, B_D_array, n, &beta_D, C_D_array, n, 10);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of this CUDA API is not supported by the Intel(R) DPC++ Compatibility Tool.
  // CHECK-NEXT: */
  cublasHgemmBatched(handle, trans3, trans3, n, n, n, &alpha_H, A_H_array, n, B_H_array, n, &beta_H, C_H_array, n, 10);

  // cublas<T>symm
  // CHECK: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(B_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::symm(*handle, (mkl::side)side0, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, m, n, dpct::get_value(&alpha_S, *handle), A_S_buf_ct{{[0-9]+}}, lda, B_S_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_S, *handle), C_S_buf_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(B_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: mkl::blas::symm(*handle, mkl::side::right, mkl::uplo::lower, m, n, dpct::get_value(&alpha_S, *handle), A_S_buf_ct{{[0-9]+}}, lda, B_S_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_S, *handle), C_S_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasSsymm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);
  cublasSsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, m, n, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto B_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::symm(*handle, (mkl::side)side1, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, m, n, dpct::get_value(&alpha_D, *handle), A_D_buf_ct{{[0-9]+}}, lda, B_D_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_D, *handle), C_D_buf_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto B_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: mkl::blas::symm(*handle, mkl::side::right, mkl::uplo::lower, m, n, dpct::get_value(&alpha_D, *handle), A_D_buf_ct{{[0-9]+}}, lda, B_D_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_D, *handle), C_D_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasDsymm(handle, (cublasSideMode_t)side1, (cublasFillMode_t)fill0, m, n, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);
  cublasDsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, m, n, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);

  // cublas<T>syrk
  // CHECK: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::syrk(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, n, k, dpct::get_value(&alpha_S, *handle), A_S_buf_ct{{[0-9]+}}, lda, dpct::get_value(&beta_S, *handle), C_S_buf_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: mkl::blas::syrk(*handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_S, *handle), A_S_buf_ct{{[0-9]+}}, lda, dpct::get_value(&beta_S, *handle), C_S_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasSsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_S, A_S, lda, &beta_S, C_S, ldc);
  cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, &beta_S, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::syrk(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, n, k, dpct::get_value(&alpha_D, *handle), A_D_buf_ct{{[0-9]+}}, lda, dpct::get_value(&beta_D, *handle), C_D_buf_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: mkl::blas::syrk(*handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_D, *handle), A_D_buf_ct{{[0-9]+}}, lda, dpct::get_value(&beta_D, *handle), C_D_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasDsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_D, A_D, lda, &beta_D, C_D, ldc);
  cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, &beta_D, C_D, ldc);

  // cublas<T>syr2k
  // CHECK: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(B_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::syr2k(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, n, k, dpct::get_value(&alpha_S, *handle), A_S_buf_ct{{[0-9]+}}, lda, B_S_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_S, *handle), C_S_buf_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(B_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: mkl::blas::syr2k(*handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_S, *handle), A_S_buf_ct{{[0-9]+}}, lda, B_S_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_S, *handle), C_S_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasSsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);
  cublasSsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto B_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::syr2k(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, n, k, dpct::get_value(&alpha_D, *handle), A_D_buf_ct{{[0-9]+}}, lda, B_D_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_D, *handle), C_D_buf_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto B_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: mkl::blas::syr2k(*handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_D, *handle), A_D_buf_ct{{[0-9]+}}, lda, B_D_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_D, *handle), C_D_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasDsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);
  cublasDsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);

  // cublas<T>trsm
  // CHECK: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::trsm(*handle, mkl::side::left, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, m, n, dpct::get_value(&alpha_S, *handle), A_S_buf_ct{{[0-9]+}}, lda, C_S_buf_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: mkl::blas::trsm(*handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, dpct::get_value(&alpha_S, *handle), A_S_buf_ct{{[0-9]+}}, lda, C_S_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasStrsm(handle, (cublasSideMode_t)0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, &alpha_S, A_S, lda, C_S, ldc);
  cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_S, A_S, lda, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::trsm(*handle, mkl::side::right, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, m, n, dpct::get_value(&alpha_D, *handle), A_D_buf_ct{{[0-9]+}}, lda, C_D_buf_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: mkl::blas::trsm(*handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, dpct::get_value(&alpha_D, *handle), A_D_buf_ct{{[0-9]+}}, lda, C_D_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasDtrsm(handle, (cublasSideMode_t)1, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, &alpha_D, A_D, lda, C_D, ldc);
  cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_D, A_D, lda, C_D, ldc);


  // CHECK: {
  // CHECK-NEXT: auto transpose_ct{{[0-9]+}} = foo();
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: mkl::blas::trsm(*handle, (mkl::side)foo(), foo()==0 ? mkl::uplo::lower : mkl::uplo::upper, (int)transpose_ct{{[0-9]+}}==2 ? mkl::transpose::conjtrans : (mkl::transpose)transpose_ct{{[0-9]+}}, (mkl::diag)foo(), m, n, dpct::get_value(&alpha_D, *handle), A_D_buf_ct{{[0-9]+}}, lda, C_D_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasDtrsm(handle, (cublasSideMode_t)foo(), (cublasFillMode_t)foo(), (cublasOperation_t)foo(), (cublasDiagType_t)foo(), m, n, &alpha_D, A_D, lda, C_D, ldc);
  return 0;
}
