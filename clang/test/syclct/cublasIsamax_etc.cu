// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/cublasIsamax_etc.sycl.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
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
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: status = (mkl::isamax(handle, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer), 0);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::isamax(handle, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: status = (mkl::idamax(handle, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer), 0);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::idamax(handle, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIdamax(handle, n, x_D, incx, result);
  cublasIdamax(handle, n, x_D, incx, result);

  //cublasI<t>amin
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: status = (mkl::isamin(handle, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer), 0);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::isamin(handle, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIsamin(handle, n, x_S, incx, result);
  cublasIsamin(handle, n, x_S, incx, result);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: status = (mkl::idamin(handle, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer), 0);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::idamin(handle, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIdamin(handle, n, x_D, incx, result);
  cublasIdamin(handle, n, x_D, incx, result);

  //cublas<t>asum
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::sasum(handle, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sasum(handle, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasSasum(handle, n, x_S, incx, result_S);
  cublasSasum(handle, n, x_S, incx, result_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dasum(handle, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dasum(handle, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasDasum(handle, n, x_D, incx, result_D);
  cublasDasum(handle, n, x_D, incx, result_D);

  //cublas<t>axpy
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::saxpy(handle, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::saxpy(handle, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSaxpy(handle, n, &alpha_S, x_S, incx, result_S, incy);
  cublasSaxpy(handle, n, &alpha_S, x_S, incx, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::daxpy(handle, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::daxpy(handle, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDaxpy(handle, n, &alpha_D, x_D, incx, result_D, incy);
  cublasDaxpy(handle, n, &alpha_D, x_D, incx, result_D, incy);

  //cublas<t>copy
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::scopy(handle, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::scopy(handle, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasScopy(handle, n, x_S, incx, result_S, incy);
  cublasScopy(handle, n, x_S, incx, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dcopy(handle, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dcopy(handle, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDcopy(handle, n, x_D, incx, result_D, incy);
  cublasDcopy(handle, n, x_D, incx, result_D, incy);

  //cublas<t>dot
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::sdot(handle, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sdot(handle, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasSdot(handle, n, x_S, incx, y_S, incy, result_S);
  cublasSdot(handle, n, x_S, incx, y_S, incy, result_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::ddot(handle, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::ddot(handle, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasDdot(handle, n, x_D, incx, y_D, incy, result_D);
  cublasDdot(handle, n, x_D, incx, y_D, incy, result_D);

  //cublas<t>nrm2
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::snrm2(handle, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::snrm2(handle, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasSnrm2(handle, n, x_S, incx, result_S);
  cublasSnrm2(handle, n, x_S, incx, result_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dnrm2(handle, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dnrm2(handle, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasDnrm2(handle, n, x_D, incx, result_D);
  cublasDnrm2(handle, n, x_D, incx, result_D);

  float *x_f = 0;
  float *y_f = 0;
  double *x_d = 0;
  double *y_d = 0;
  //cublas<t>rot
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::srot(handle, n, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, *(x_S), *(y_S)), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::srot(handle, n, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, *(x_S), *(y_S));
  // CHECK-NEXT: }
  status = cublasSrot(handle, n, x_f, incx, y_f, incy, x_S, y_S);
  cublasSrot(handle, n, x_f, incx, y_f, incy, x_S, y_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::drot(handle, n, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, *(x_D), *(y_D)), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::drot(handle, n, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, *(x_D), *(y_D));
  // CHECK-NEXT: }
  status = cublasDrot(handle, n, x_d, incx, y_d, incy, x_D, y_D);
  cublasDrot(handle, n, x_d, incx, y_d, incy, x_D, y_D);

  //cublas<t>rotg
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::srotg(handle, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::srotg(handle, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasSrotg(handle, x_f, y_f, x_f, y_f);
  cublasSrotg(handle, x_f, y_f, x_f, y_f);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::drotg(handle, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::drotg(handle, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasDrotg(handle, x_d, y_d, x_d, y_d);
  cublasDrotg(handle, x_d, y_d, x_d, y_d);

  //cublas<t>rotm
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::srotm(handle, n, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::srotm(handle, n, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasSrotm(handle, n, x_f, incx, y_f, incy, x_S);
  cublasSrotm(handle, n, x_f, incx, y_f, incy, x_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::drotm(handle, n, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::drotm(handle, n, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasDrotm(handle, n, x_d, incx, y_d, incy, x_D);
  cublasDrotm(handle, n, x_d, incx, y_d, incy, x_D);

  //cublas<t>rotmg
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::srotmg(handle, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, *(x_S), y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::srotmg(handle, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, *(x_S), y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasSrotmg(handle, x_f, y_f, y_f, x_S, y_f);
  cublasSrotmg(handle, x_f, y_f, y_f, x_S, y_f);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::drotmg(handle, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, *(x_D), y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::drotmg(handle, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, *(x_D), y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasDrotmg(handle, x_d, y_d, y_d, x_D, y_d);
  cublasDrotmg(handle, x_d, y_d, y_d, x_D, y_d);

  //cublas<t>scal
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::sscal(handle, n, *(&alpha_S), x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sscal(handle, n, *(&alpha_S), x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasSscal(handle, n, &alpha_S, x_f, incx);
  cublasSscal(handle, n, &alpha_S, x_f, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dscal(handle, n, *(&alpha_D), x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dscal(handle, n, *(&alpha_D), x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasDscal(handle, n, &alpha_D, x_d, incx);
  cublasDscal(handle, n, &alpha_D, x_d, incx);

  //cublas<t>swap
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::sswap(handle, n, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sswap(handle, n, x_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSswap(handle, n, x_f, incx, y_f, incy);
  cublasSswap(handle, n, x_f, incx, y_f, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dswap(handle, n, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dswap(handle, n, x_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDswap(handle, n, x_d, incx, y_d, incy);
  cublasDswap(handle, n, x_d, incx, y_d, incy);

  //level2
  //cublas<t>gbmv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::sgbmv(handle, mkl::transpose::nontrans, m, n, m, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_S), result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sgbmv(handle, mkl::transpose::nontrans, m, n, m, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_S), result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dgbmv(handle, mkl::transpose::nontrans, m, n, m, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_D), result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dgbmv(handle, mkl::transpose::nontrans, m, n, m, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_D), result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>gemv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::sgemv(handle, mkl::transpose::nontrans, m, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_S), result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sgemv(handle, mkl::transpose::nontrans, m, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_S), result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dgemv(handle, mkl::transpose::nontrans, m, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_D), result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dgemv(handle, mkl::transpose::nontrans, m, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_D), result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>ger
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::sger(handle, m, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sger(handle, m, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasSger(handle, m, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  cublasSger(handle, m, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dger(handle, m, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dger(handle, m, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasDger(handle, m, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  cublasDger(handle, m, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);

  //cublas<t>sbmv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::ssbmv(handle, mkl::uplo::upper, m, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_S), result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssbmv(handle, mkl::uplo::upper, m, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_S), result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dsbmv(handle, mkl::uplo::upper, m, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_D), result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsbmv(handle, mkl::uplo::upper, m, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_D), result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>spmv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::sspmv(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_S), result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sspmv(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_S), result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, y_S, incx, &beta_S, result_S, incy);
  cublasSspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, y_S, incx, &beta_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dspmv(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_D), result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dspmv(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_D), result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, y_D, incx, &beta_D, result_D, incy);
  cublasDspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>spr
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::sspr(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sspr(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasSspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S);
  cublasSspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dspr(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dspr(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasDspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D);
  cublasDspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D);

  //cublas<t>spr2
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::sspr2(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sspr2(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasSspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S);
  cublasSspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dspr2(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dspr2(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasDspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D);
  cublasDspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D);

  //cublas<t>symv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::ssymv(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_S), result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssymv(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_S), result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dsymv(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_D), result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsymv(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, *(&beta_D), result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>syr
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::ssyr(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssyr(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasSsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S, lda);
  cublasSsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S, lda);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dsyr(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsyr(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasDsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D, lda);
  cublasDsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D, lda);

  //cublas<t>syr2
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::ssyr2(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(y_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssyr2(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasSsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  cublasSsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dsyr2(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(y_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsyr2(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasDsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  cublasDsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);

  //cublas<t>tbmv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::stbmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::stbmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);
  cublasStbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dtbmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtbmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);
  cublasDtbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);

  //cublas<t>tbsv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::stbsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::stbsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);
  cublasStbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dtbsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtbsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);
  cublasDtbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);

  //cublas<t>tpmv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::stpmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::stpmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);
  cublasStpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dtpmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtpmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);
  cublasDtpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);

  //cublas<t>tpsv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::stpsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::stpsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);
  cublasStpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dtpsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtpsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);
  cublasDtpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);

  //cublas<t>trmv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::strmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::strmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);
  cublasStrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dtrmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtrmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, lda, result_D, incy);
  cublasDtrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, lda, result_D, incy);

  //cublas<t>trsv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::strsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(x_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::strsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasStrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);
  cublasStrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dtrsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(x_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtrsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, lda, result_D, incy);
  cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, lda, result_D, incy);

  //level3

  // cublas<T>symm
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::ssymm(handle, mkl::side::left, mkl::uplo::upper, m, n, *(&alpha_S), A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_S), C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssymm(handle, mkl::side::right, mkl::uplo::lower, m, n, *(&alpha_S), A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_S), C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasSsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);
  cublasSsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, m, n, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dsymm(handle, mkl::side::left, mkl::uplo::upper, m, n, *(&alpha_D), A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_D), C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsymm(handle, mkl::side::right, mkl::uplo::lower, m, n, *(&alpha_D), A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_D), C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasDsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);
  cublasDsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, m, n, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);

  // cublas<T>syrk
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::ssyrk(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_S), A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, *(&beta_S), C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssyrk(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_S), A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, *(&beta_S), C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, &beta_S, C_S, ldc);
  cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, &beta_S, C_S, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dsyrk(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_D), A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, *(&beta_D), C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsyrk(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_D), A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, *(&beta_D), C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, &beta_D, C_D, ldc);
  cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, &beta_D, C_D, ldc);

  // cublas<T>syr2k
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::ssyr2k(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_S), A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_S), C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::ssyr2k(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_S), A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_S), C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasSsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);
  cublasSsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dsyr2k(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_D), A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_D), C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(B_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dsyr2k(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_D), A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_D), C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasDsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);
  cublasDsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);

  // cublas<T>trsm
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::strsm(handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, m, n, *(&alpha_S), A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::strsm(handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, *(&alpha_S), A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, n, &alpha_S, A_S, lda, C_S, ldc);
  cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_S, A_S, lda, C_S, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dtrsm(handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, m, n, *(&alpha_D), A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_D);
  // CHECK-NEXT: cl::sycl::buffer<double,1> C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_D_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dtrsm(handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, *(&alpha_D), A_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, C_D_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, n, &alpha_D, A_D, lda, C_D, ldc);
  cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_D, A_D, lda, C_D, ldc);

  return 0;
}
