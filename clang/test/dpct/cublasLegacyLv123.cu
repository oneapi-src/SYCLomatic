// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasLegacyLv123.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas.h>
#include <cuda_runtime.h>
int main() {
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
  //level1

  //cublasI<t>amax
  // CHECK: int res;
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamax(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: res = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  int res = cublasIsamax(n, x_S, incx);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamax(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIdamax(n, x_D, incx);

  //cublasI<t>amin
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamin(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIsamin(n, x_S, incx);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamin(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIdamin(n, x_D, incx);

  //cublas<t>asum
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::asum(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_S = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasSasum(n, x_S, incx);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::asum(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_D = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDasum(n, x_D, incx);

  //cublas<t>dot
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::dot(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, result_temp_buffer);
  // CHECK-NEXT: *result_S = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasSdot(n, x_S, incx, y_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::dot(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, result_temp_buffer);
  // CHECK-NEXT: *result_D = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDdot(n, x_D, incx, y_D, incy);

  //cublas<t>nrm2
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::nrm2(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_S = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasSnrm2(n, x_S, incx);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::nrm2(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_D = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDnrm2(n, x_D, incx);




  //cublas<t>axpy
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::axpy(dpct::get_default_queue(), n, alpha_S, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSaxpy(n, alpha_S, x_S, incx, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::axpy(dpct::get_default_queue(), n, alpha_D, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDaxpy(n, alpha_D, x_D, incx, result_D, incy);

  //cublas<t>copy
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::copy(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasScopy(n, x_S, incx, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::copy(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDcopy(n, x_D, incx, result_D, incy);


  //cublas<t>rot
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::rot(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, *x_S, *y_S);
  // CHECK-NEXT: }
  cublasSrot(n, x_f, incx, y_f, incy, *x_S, *y_S);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::rot(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, *x_D, *y_D);
  // CHECK-NEXT: }
  cublasDrot(n, x_d, incx, y_d, incy, *x_D, *y_D);

  //cublas<t>rotg
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::rotg(dpct::get_default_queue(), buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasSrotg(x_f, y_f, x_f, y_f);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::rotg(dpct::get_default_queue(), buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasDrotg(x_d, y_d, x_d, y_d);

  //cublas<t>rotm
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::rotm(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasSrotm(n, x_f, incx, y_f, incy, x_S);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::rotm(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasDrotm(n, x_d, incx, y_d, incy, x_D);

  //cublas<t>rotmg
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::rotmg(dpct::get_default_queue(), buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, *(x_S), buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasSrotmg(x_f, y_f, y_f, x_S, y_f);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::rotmg(dpct::get_default_queue(), buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, *(x_D), buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasDrotmg(x_d, y_d, y_d, x_D, y_d);

  //cublas<t>scal
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::scal(dpct::get_default_queue(), n, alpha_S, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  cublasSscal(n, alpha_S, x_f, incx);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::scal(dpct::get_default_queue(), n, alpha_D, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  cublasDscal(n, alpha_D, x_d, incx);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::swap(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSswap(n, x_f, incx, y_f, incy);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::swap(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDswap(n, x_d, incx, y_d, incy);

  //level2
  //cublas<t>gbmv
  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::gbmv(dpct::get_default_queue(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, m, n, alpha_S, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, beta_S, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSgbmv('N', m, n, m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::gbmv(dpct::get_default_queue(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, m, n, alpha_D, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, beta_D, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDgbmv( 'N', m, n, m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>gemv
  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemv(dpct::get_default_queue(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, alpha_S, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, beta_S, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSgemv('N', m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::gemv(dpct::get_default_queue(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, alpha_D, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, beta_D, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDgemv('N', m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>ger
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::ger(dpct::get_default_queue(), m, n, alpha_S, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasSger(m, n, alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::ger(dpct::get_default_queue(), m, n, alpha_D, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasDger(m, n, alpha_D, x_D, incx, y_D, incy, result_D, lda);

  //cublas<t>sbmv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::sbmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, alpha_S, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, beta_S, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSsbmv('U', m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::sbmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, alpha_D, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, beta_D, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDsbmv('U', m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>spmv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::spmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx, beta_S, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSspmv('U', n, alpha_S, x_S, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::spmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx, beta_D, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDspmv('U', n, alpha_D, x_D, y_D, incx, beta_D, result_D, incy);

  //cublas<t>spr
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::spr(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasSspr('U', n, alpha_S, x_S, incx, result_S);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::spr(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasDspr('U', n, alpha_D, x_D, incx, result_D);

  //cublas<t>spr2
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::spr2(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasSspr2('U', n, alpha_S, x_S, incx, y_S, incy, result_S);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::spr2(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasDspr2('U', n, alpha_D, x_D, incx, y_D, incy, result_D);

  //cublas<t>symv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::symv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, beta_S, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasSsymv('U', n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::symv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, beta_D, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDsymv('U', n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>syr
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::syr(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasSsyr('U', n, alpha_S, x_S, incx, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::syr(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasDsyr('U', n, alpha_D, x_D, incx, result_D, lda);

  //cublas<t>syr2
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::syr2(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasSsyr2('U', n, alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::syr2(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasDsyr2('U', n, alpha_D, x_D, incx, y_D, incy, result_D, lda);

  //cublas<t>tbmv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::tbmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasStbmv('U', 'N', 'U', n, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'u';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'u';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::tbmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDtbmv('u', 'N', 'u', n, n, x_D, lda, result_D, incy);

  //cublas<t>tbsv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'L';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::tbsv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasStbsv('L', 'N', 'U', n, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'l';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::tbsv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDtbsv('l', 'N', 'U', n, n, x_D, lda, result_D, incy);

  //cublas<t>tpmv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::tpmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasStpmv('U', 'N', 'U', n, x_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::tpmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDtpmv('U', 'N', 'U', n, x_D, result_D, incy);

  //cublas<t>tpsv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::tpsv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasStpsv('U', 'N', 'U', n, x_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::tpsv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDtpsv('U', 'N', 'U', n, x_D, result_D, incy);

  //cublas<t>trmv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::trmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasStrmv('U', 'N', 'U', n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::trmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDtrmv('U', 'N', 'U', n, x_D, lda, result_D, incy);

  //cublas<t>trsv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::trsv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasStrsv('U', 'N', 'U', n, x_S, lda, result_S, incy);


  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::trsv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasDtrsv('U', 'N', 'U', n, x_D, lda, result_D, incy);

  //level3

  // cublas<T>symm
  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'R';
  // CHECK-NEXT: auto fillmode_ct1 = 'L';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::symm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, alpha_S, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, beta_S, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasSsymm('R', 'L', m, n, alpha_S, A_S, lda, B_S, ldb, beta_S, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'r';
  // CHECK-NEXT: auto fillmode_ct1 = 'L';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::symm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, alpha_D, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, beta_D, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasDsymm('r', 'L', m, n, alpha_D, A_D, lda, B_D, ldb, beta_D, C_D, ldc);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'T';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::syrk(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, alpha_S, buffer_ct{{[0-9]+}}, lda, beta_S, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasSsyrk('U', 'T', n, k, alpha_S, A_S, lda, beta_S, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 't';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::syrk(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, alpha_D, buffer_ct{{[0-9]+}}, lda, beta_D, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasDsyrk('U', 't', n, k, alpha_D, A_D, lda, beta_D, C_D, ldc);

  // cublas<T>syr2k
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'C';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::syr2k(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, alpha_S, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, beta_S, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasSsyr2k('U', 'C', n, k, alpha_S, A_S, lda, B_S, ldb, beta_S, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'c';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::syr2k(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, alpha_D, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, beta_D, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasDsyr2k('U', 'c', n, k, alpha_D, A_D, lda, B_D, ldb, beta_D, C_D, ldc);

  // cublas<T>trsm
  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'L';
  // CHECK-NEXT: auto fillmode_ct1 = 'U';
  // CHECK-NEXT: auto transpose_ct2 = 'N';
  // CHECK-NEXT: auto diagtype_ct3 = 'n';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::trsm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct2)=='N'||(transpose_ct2)=='n')?(mkl::transpose::nontrans):(((transpose_ct2)=='T'||(transpose_ct2)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct3)=='N'||(diagtype_ct3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, alpha_S, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasStrsm('L', 'U', 'N', 'n', m, n, alpha_S, A_S, lda, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'l';
  // CHECK-NEXT: auto fillmode_ct1 = 'U';
  // CHECK-NEXT: auto transpose_ct2 = 'N';
  // CHECK-NEXT: auto diagtype_ct3 = 'N';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::trsm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct2)=='N'||(transpose_ct2)=='n')?(mkl::transpose::nontrans):(((transpose_ct2)=='T'||(transpose_ct2)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct3)=='N'||(diagtype_ct3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, alpha_D, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasDtrsm('l', 'U', 'N', 'N', m, n, alpha_D, A_D, lda, C_D, ldc);

  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'T';
  // CHECK-NEXT: auto transpose_ct1 = 'C';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(dpct::get_default_queue(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, n, n, alpha_S, buffer_ct{{[0-9]+}}, n, buffer_ct{{[0-9]+}}, n, beta_S, buffer_ct{{[0-9]+}}, n);
  // CHECK-NEXT: }
  cublasSgemm('T', 'C', n, n, n, alpha_S, A_S, n, B_S, n, beta_S, C_S, n);

  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: auto transpose_ct1 = 'n';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::gemm(dpct::get_default_queue(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, n, n, alpha_D, buffer_ct{{[0-9]+}}, n, buffer_ct{{[0-9]+}}, n, beta_D, buffer_ct{{[0-9]+}}, n);
  // CHECK-NEXT: }
  cublasDgemm('N', 'n', n, n, n, alpha_D, A_D, n, B_D, n, beta_D, C_D, n);


}
