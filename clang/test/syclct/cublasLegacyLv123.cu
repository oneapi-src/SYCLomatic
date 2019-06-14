// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/cublasLegacyLv123.sycl.cpp --match-full-lines %s
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
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::isamax(syclct::get_default_queue(), n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: res = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  int res = cublasIsamax(n, x_S, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::idamax(syclct::get_default_queue(), n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIdamax(n, x_D, incx);

  //cublasI<t>amin
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::isamin(syclct::get_default_queue(), n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIsamin(n, x_S, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::idamin(syclct::get_default_queue(), n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIdamin(n, x_D, incx);

  //cublas<t>asum
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::sasum(syclct::get_default_queue(), n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_S = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasSasum(n, x_S, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::dasum(syclct::get_default_queue(), n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_D = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDasum(n, x_D, incx);

  //cublas<t>dot
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::sdot(syclct::get_default_queue(), n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_temp_buffer);
  // CHECK-NEXT: *result_S = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasSdot(n, x_S, incx, y_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::ddot(syclct::get_default_queue(), n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_temp_buffer);
  // CHECK-NEXT: *result_D = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDdot(n, x_D, incx, y_D, incy);

  //cublas<t>nrm2
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::snrm2(syclct::get_default_queue(), n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_S = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasSnrm2(n, x_S, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::dnrm2(syclct::get_default_queue(), n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_D = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDnrm2(n, x_D, incx);




  //cublas<t>axpy
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::saxpy(syclct::get_default_queue(), n, alpha_S, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasSaxpy(n, alpha_S, x_S, incx, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::daxpy(syclct::get_default_queue(), n, alpha_D, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDaxpy(n, alpha_D, x_D, incx, result_D, incy);

  //cublas<t>copy
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::scopy(syclct::get_default_queue(), n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasScopy(n, x_S, incx, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dcopy(syclct::get_default_queue(), n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDcopy(n, x_D, incx, result_D, incy);


  //cublas<t>rot
  // CHECK: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::srot(syclct::get_default_queue(), n, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, *x_S, *y_S);
  // CHECK-NEXT: }
  cublasSrot(n, x_f, incx, y_f, incy, *x_S, *y_S);

  // CHECK: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::drot(syclct::get_default_queue(), n, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, *x_D, *y_D);
  // CHECK-NEXT: }
  cublasDrot(n, x_d, incx, y_d, incy, *x_D, *y_D);

  //cublas<t>rotg
  // CHECK: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::srotg(syclct::get_default_queue(), x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasSrotg(x_f, y_f, x_f, y_f);

  // CHECK: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::drotg(syclct::get_default_queue(), x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasDrotg(x_d, y_d, x_d, y_d);

  //cublas<t>rotm
  // CHECK: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::srotm(syclct::get_default_queue(), n, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasSrotm(n, x_f, incx, y_f, incy, x_S);

  // CHECK: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::drotm(syclct::get_default_queue(), n, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasDrotm(n, x_d, incx, y_d, incy, x_D);

  //cublas<t>rotmg
  // CHECK: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::srotmg(syclct::get_default_queue(), x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, *(x_S), y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasSrotmg(x_f, y_f, y_f, x_S, y_f);

  // CHECK: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::drotmg(syclct::get_default_queue(), x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, *(x_D), y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasDrotmg(x_d, y_d, y_d, x_D, y_d);

  //cublas<t>scal
  // CHECK: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sscal(syclct::get_default_queue(), n, alpha_S, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  cublasSscal(n, alpha_S, x_f, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dscal(syclct::get_default_queue(), n, alpha_D, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  cublasDscal(n, alpha_D, x_d, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sswap(syclct::get_default_queue(), n, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasSswap(n, x_f, incx, y_f, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dswap(syclct::get_default_queue(), n, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDswap(n, x_d, incx, y_d, incy);

  //level2
  //cublas<t>gbmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sgbmv(syclct::get_default_queue(), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, m, n, alpha_S, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, beta_S, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasSgbmv('N', m, n, m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dgbmv(syclct::get_default_queue(), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, m, n, alpha_D, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, beta_D, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDgbmv( 'N', m, n, m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>gemv
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sgemv(syclct::get_default_queue(), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, alpha_S, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, beta_S, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasSgemv('N', m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dgemv(syclct::get_default_queue(), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, alpha_D, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, beta_D, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDgemv('N', m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>ger
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sger(syclct::get_default_queue(), m, n, alpha_S, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasSger(m, n, alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dger(syclct::get_default_queue(), m, n, alpha_D, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasDger(m, n, alpha_D, x_D, incx, y_D, incy, result_D, lda);

  //cublas<t>sbmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssbmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, alpha_S, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, beta_S, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasSsbmv('U', m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsbmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, alpha_D, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, beta_D, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDsbmv('U', m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>spmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sspmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, beta_S, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasSspmv('U', n, alpha_S, x_S, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dspmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, beta_D, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDspmv('U', n, alpha_D, x_D, y_D, incx, beta_D, result_D, incy);

  //cublas<t>spr
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sspr(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasSspr('U', n, alpha_S, x_S, incx, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dspr(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasDspr('U', n, alpha_D, x_D, incx, result_D);

  //cublas<t>spr2
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sspr2(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasSspr2('U', n, alpha_S, x_S, incx, y_S, incy, result_S);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dspr2(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasDspr2('U', n, alpha_D, x_D, incx, y_D, incy, result_D);

  //cublas<t>symv
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssymv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, beta_S, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasSsymv('U', n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsymv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, beta_D, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDsymv('U', n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>syr
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssyr(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasSsyr('U', n, alpha_S, x_S, incx, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsyr(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasDsyr('U', n, alpha_D, x_D, incx, result_D, lda);

  //cublas<t>syr2
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssyr2(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasSsyr2('U', n, alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsyr2(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasDsyr2('U', n, alpha_D, x_D, incx, y_D, incy, result_D, lda);

  //cublas<t>tbmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::stbmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasStbmv('U', 'N', 'U', n, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtbmv(syclct::get_default_queue(), ((('u')=='L'||('u')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('u')=='N'||('u')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDtbmv('u', 'N', 'u', n, n, x_D, lda, result_D, incy);

  //cublas<t>tbsv
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::stbsv(syclct::get_default_queue(), ((('L')=='L'||('L')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasStbsv('L', 'N', 'U', n, n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtbsv(syclct::get_default_queue(), ((('l')=='L'||('l')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDtbsv('l', 'N', 'U', n, n, x_D, lda, result_D, incy);

  //cublas<t>tpmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::stpmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasStpmv('U', 'N', 'U', n, x_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtpmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDtpmv('U', 'N', 'U', n, x_D, result_D, incy);

  //cublas<t>tpsv
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::stpsv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasStpsv('U', 'N', 'U', n, x_S, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtpsv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDtpsv('U', 'N', 'U', n, x_D, result_D, incy);

  //cublas<t>trmv
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::strmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasStrmv('U', 'N', 'U', n, x_S, lda, result_S, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtrmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDtrmv('U', 'N', 'U', n, x_D, lda, result_D, incy);

  //cublas<t>trsv
  // CHECK: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::strsv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasStrsv('U', 'N', 'U', n, x_S, lda, result_S, incy);


  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtrsv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasDtrsv('U', 'N', 'U', n, x_D, lda, result_D, incy);

  //level3

  // cublas<T>symm
  // CHECK: {
  // CHECK-NEXT: auto A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssymm(syclct::get_default_queue(), ((('R')=='L'||('R')=='l')?(mkl::side::left):(mkl::side::right)), ((('L')=='L'||('L')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, alpha_S, A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, beta_S, C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasSsymm('R', 'L', m, n, alpha_S, A_S, lda, B_S, ldb, beta_S, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsymm(syclct::get_default_queue(), ((('r')=='L'||('r')=='l')?(mkl::side::left):(mkl::side::right)), ((('L')=='L'||('L')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, alpha_D, A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, beta_D, C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasDsymm('r', 'L', m, n, alpha_D, A_D, lda, B_D, ldb, beta_D, C_D, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssyrk(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('T')=='N'||('T')=='n')?(mkl::transpose::nontrans):((('T')=='T'||('T')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, alpha_S, A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, beta_S, C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasSsyrk('U', 'T', n, k, alpha_S, A_S, lda, beta_S, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsyrk(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('t')=='N'||('t')=='n')?(mkl::transpose::nontrans):((('t')=='T'||('t')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, alpha_D, A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, beta_D, C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasDsyrk('U', 't', n, k, alpha_D, A_D, lda, beta_D, C_D, ldc);

  // cublas<T>syr2k
  // CHECK: {
  // CHECK-NEXT: auto A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssyr2k(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('C')=='N'||('C')=='n')?(mkl::transpose::nontrans):((('C')=='T'||('C')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, alpha_S, A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, beta_S, C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasSsyr2k('U', 'C', n, k, alpha_S, A_S, lda, B_S, ldb, beta_S, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsyr2k(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('c')=='N'||('c')=='n')?(mkl::transpose::nontrans):((('c')=='T'||('c')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, alpha_D, A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, beta_D, C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasDsyr2k('U', 'c', n, k, alpha_D, A_D, lda, B_D, ldb, beta_D, C_D, ldc);

  // cublas<T>trsm
  // CHECK: {
  // CHECK-NEXT: auto A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::strsm(syclct::get_default_queue(), ((('L')=='L'||('L')=='l')?(mkl::side::left):(mkl::side::right)), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('n')=='N'||('n')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, alpha_S, A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasStrsm('L', 'U', 'N', 'n', m, n, alpha_S, A_S, lda, C_S, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtrsm(syclct::get_default_queue(), ((('l')=='L'||('l')=='l')?(mkl::side::left):(mkl::side::right)), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('N')=='N'||('N')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, alpha_D, A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasDtrsm('l', 'U', 'N', 'N', m, n, alpha_D, A_D, lda, C_D, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sgemm(syclct::get_default_queue(), ((('T')=='N'||('T')=='n')?(mkl::transpose::nontrans):((('T')=='T'||('T')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('C')=='N'||('C')=='n')?(mkl::transpose::nontrans):((('C')=='T'||('C')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, n, n, alpha_S, A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, n, B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, n, beta_S, C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, n);
  // CHECK-NEXT: }
  cublasSgemm('T', 'C', n, n, n, alpha_S, A_S, n, B_S, n, beta_S, C_S, n);

  // CHECK: {
  // CHECK-NEXT: auto A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dgemm(syclct::get_default_queue(), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('n')=='N'||('n')=='n')?(mkl::transpose::nontrans):((('n')=='T'||('n')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, n, n, alpha_D, A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, n, B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, n, beta_D, C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, n);
  // CHECK-NEXT: }
  cublasDgemm('N', 'n', n, n, n, alpha_D, A_D, n, B_D, n, beta_D, C_D, n);


}
