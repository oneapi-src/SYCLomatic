// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/cublasRegularCZ.sycl.cpp --match-full-lines %s

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
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: status = (mkl::icamax(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer), 0);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::icamax(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIcamax(handle, n, x_c, incx, result);
  cublasIcamax(handle, n, x_c, incx, result);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: status = (mkl::izamax(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer), 0);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::izamax(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIzamax(handle, n, x_z, incx, result);
  cublasIzamax(handle, n, x_z, incx, result);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: status = (mkl::icamin(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer), 0);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::icamin(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIcamin(handle, n, x_c, incx, result);
  cublasIcamin(handle, n, x_c, incx, result);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: status = (mkl::izamin(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer), 0);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int,1> result_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<int, 1>(cl::sycl::range<1>(result_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::izamin(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: result_{{[0-9]+}}_buffer_{{[0-9a-z]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIzamin(handle, n, x_z, incx, result);
  cublasIzamin(handle, n, x_z, incx, result);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::scasum(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::scasum(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasScasum(handle, n, x_c, incx, result_f);
  cublasScasum(handle, n, x_c, incx, result_f);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dzasum(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dzasum(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
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

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::caxpy(handle, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::caxpy(handle, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCaxpy(handle, n, alpha_c, x_c, incx, y_c, incy);
  cublasCaxpy(handle, n, alpha_c, x_c, incx, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zaxpy(handle, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zaxpy(handle, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZaxpy(handle, n, alpha_z, x_z, incx, y_z, incy);
  cublasZaxpy(handle, n, alpha_z, x_z, incx, y_z, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::ccopy(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ccopy(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCcopy(handle, n, x_c, incx, y_c, incy);
  cublasCcopy(handle, n, x_c, incx, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zcopy(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zcopy(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZcopy(handle, n, x_z, incx, y_z, incy);
  cublasZcopy(handle, n, x_z, incx, y_z, incy);

  cuComplex* result_c = 0;
  cuDoubleComplex* result_z = 0;

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cdotu(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cdotu(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasCdotu(handle, n, x_c, incx, y_c, incy, result_c);
  cublasCdotu(handle, n, x_c, incx, y_c, incy, result_c);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cdotc(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cdotc(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasCdotc(handle, n, x_c, incx, y_c, incy, result_c);
  cublasCdotc(handle, n, x_c, incx, y_c, incy, result_c);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zdotu(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zdotu(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasZdotu(handle, n, x_z, incx, y_z, incy, result_z);
  cublasZdotu(handle, n, x_z, incx, y_z, incy, result_z);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zdotc(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zdotc(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasZdotc(handle, n, x_z, incx, y_z, incy, result_z);
  cublasZdotc(handle, n, x_z, incx, y_z, incy, result_z);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::scnrm2(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(result_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::scnrm2(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasScnrm2(handle, n, x_c, incx, result_f);
  cublasScnrm2(handle, n, x_c, incx, result_f);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::dznrm2(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(result_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::dznrm2(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
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

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::csrot(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, *(c_f), *(s_f)), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::csrot(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, *(c_f), *(s_f));
  // CHECK-NEXT: }
  status = cublasCsrot(handle, n, x_c, incx, y_c, incy, c_f, s_f);
  cublasCsrot(handle, n, x_c, incx, y_c, incy, c_f, s_f);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zdrot(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, *(c_d), *(s_d)), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zdrot(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, *(c_d), *(s_d));
  // CHECK-NEXT: }
  status = cublasZdrot(handle, n, x_z, incx, y_z, incy, c_d, s_d);
  cublasZdrot(handle, n, x_z, incx, y_z, incy, c_d, s_d);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto c_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(c_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> c_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = c_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(c_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto s_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(s_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> s_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = s_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(s_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::crotg(handle, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, c_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, s_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto c_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(c_f);
  // CHECK-NEXT: cl::sycl::buffer<float,1> c_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = c_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(c_f_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto s_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(s_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> s_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = s_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(s_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::crotg(handle, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, c_f_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, s_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasCrotg(handle, x_c, y_c, c_f, s_c);
  cublasCrotg(handle, x_c, y_c, c_f, s_c);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto c_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(c_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> c_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = c_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(c_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto s_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(s_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> s_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = s_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(s_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zrotg(handle, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, c_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, s_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto c_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(c_d);
  // CHECK-NEXT: cl::sycl::buffer<double,1> c_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = c_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(c_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto s_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(s_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> s_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = s_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(s_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zrotg(handle, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, c_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, s_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasZrotg(handle, x_z, y_z, c_d, s_z);
  cublasZrotg(handle, x_z, y_z, c_d, s_z);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cscal(handle, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cscal(handle, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCscal(handle, n, alpha_c, x_c, incx);
  cublasCscal(handle, n, alpha_c, x_c, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zscal(handle, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zscal(handle, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZscal(handle, n, alpha_z, x_z, incx);
  cublasZscal(handle, n, alpha_z, x_z, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::csscal(handle, n, *(alpha_f), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::csscal(handle, n, *(alpha_f), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCsscal(handle, n, alpha_f, x_c, incx);
  cublasCsscal(handle, n, alpha_f, x_c, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zdscal(handle, n, *(alpha_d), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zdscal(handle, n, *(alpha_d), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZdscal(handle, n, alpha_d, x_z, incx);
  cublasZdscal(handle, n, alpha_d, x_z, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cswap(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cswap(handle, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCswap(handle, n, x_c, incx, y_c, incy);
  cublasCswap(handle, n, x_c, incx, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zswap(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zswap(handle, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZswap(handle, n, x_z, incx, y_z, incy);
  cublasZswap(handle, n, x_z, incx, y_z, incy);

  //level 2
  int m=0;
  int kl=0;
  int ku=0;
  int lda = 10;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cgbmv(handle, mkl::transpose::nontrans, m, n, kl, ku, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cgbmv(handle, mkl::transpose::nontrans, m, n, kl, ku, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasCgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zgbmv(handle, mkl::transpose::nontrans, m, n, kl, ku, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zgbmv(handle, mkl::transpose::nontrans, m, n, kl, ku, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cgemv(handle, mkl::transpose::nontrans, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cgemv(handle, mkl::transpose::nontrans, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCgemv(handle, CUBLAS_OP_N, m, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasCgemv(handle, CUBLAS_OP_N, m, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zgemv(handle, mkl::transpose::nontrans, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zgemv(handle, mkl::transpose::nontrans, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZgemv(handle, CUBLAS_OP_N, m, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZgemv(handle, CUBLAS_OP_N, m, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cgeru(handle, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cgeru(handle, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCgeru(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCgeru(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cgerc(handle, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cgerc(handle, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCgerc(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCgerc(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zgeru(handle, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zgeru(handle, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZgeru(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZgeru(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zgerc(handle, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zgerc(handle, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZgerc(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZgerc(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  int k = 1;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::ctbmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, k, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctbmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, k, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_c, lda, result_c, incx);
  cublasCtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_c, lda, result_c, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::ztbmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, k, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztbmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, k, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_z, lda, result_z, incx);
  cublasZtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_z, lda, result_z, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::ctbsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit,  n, k, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctbsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit,  n, k, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_c, lda, result_c, incx);
  cublasCtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_c, lda, result_c, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::ztbsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit,  n, k, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztbsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit,  n, k, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_z, lda, result_z, incx);
  cublasZtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_z, lda, result_z, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::ctpmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctpmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);
  cublasCtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::ztpmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztpmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);
  cublasZtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::ctpsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctpsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);
  cublasCtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::ztpsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztpsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);
  cublasZtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::ctrmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctrmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);
  cublasCtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::ztrmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztrmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);
  cublasZtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::ctrsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctrsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);
  cublasCtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::ztrsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztrsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);
  cublasZtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::chemv(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::chemv(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasChemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasChemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zhemv(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zhemv(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZhemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZhemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::chbmv(handle, mkl::uplo::lower, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::chbmv(handle, mkl::uplo::lower, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasChbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasChbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zhbmv(handle, mkl::uplo::lower, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zhbmv(handle, mkl::uplo::lower, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZhbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZhbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::chpmv(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::chpmv(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasChpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, x_c, incx, beta_c, y_c, incy);
  cublasChpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, x_c, incx, beta_c, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zhpmv(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zhpmv(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZhpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, x_z, incx, beta_z, y_z, incy);
  cublasZhpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, x_z, incx, beta_z, y_z, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cher(handle, mkl::uplo::lower, n, *(alpha_f), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cher(handle, mkl::uplo::lower, n, *(alpha_f), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c, lda);
  cublasCher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c, lda);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zher(handle, mkl::uplo::lower, n, *(alpha_d), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zher(handle, mkl::uplo::lower, n, *(alpha_d), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z, lda);
  cublasZher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z, lda);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cher2(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cher2(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zher2(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zher2(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::chpr(handle, mkl::uplo::lower, n, *(alpha_f), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::chpr(handle, mkl::uplo::lower, n, *(alpha_f), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasChpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c);
  cublasChpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zhpr(handle, mkl::uplo::lower, n, *(alpha_d), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zhpr(handle, mkl::uplo::lower, n, *(alpha_d), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasZhpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z);
  cublasZhpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::chpr2(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::chpr2(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasChpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c);
  cublasChpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zhpr2(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zhpr2(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  status = cublasZhpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z);
  cublasZhpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z);

  int N = 100;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N, std::complex<float>((beta_c)->x(),(beta_c)->y()), result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), x_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N, y_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N, std::complex<float>((beta_c)->x(),(beta_c)->y()), result_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N);
  // CHECK-NEXT: }
  status = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha_c, x_c, N, y_c, N, beta_c, result_c, N);
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha_c, x_c, N, y_c, N, beta_c, result_c, N);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N, std::complex<double>((beta_z)->x(),(beta_z)->y()), result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), x_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N, y_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N, std::complex<double>((beta_z)->x(),(beta_z)->y()), result_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N);
  // CHECK-NEXT: }
  status = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha_z, x_z, N, y_z, N, beta_z, result_z, N);
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

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:62: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cgemm3m(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, m, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cgemm3m(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, m, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:63: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zgemm3m(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, m, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zgemm3m(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, m, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:64: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::csymm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::csymm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:65: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zsymm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zsymm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:66: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::csyrk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::csyrk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);
  cublasCsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:67: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zsyrk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zsyrk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);
  cublasZsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:68: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::csyr2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::csyr2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:69: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zsyr2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zsyr2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:70: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::ctrsm(handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctrsm(handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb);
  // CHECK-NEXT: }
  status = cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_c, A_c, lda, B_c, ldb);
  cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_c, A_c, lda, B_c, ldb);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:71: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::ztrsm(handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztrsm(handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb);
  // CHECK-NEXT: }
  status = cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_z, A_z, lda, B_z, ldb);
  cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_z, A_z, lda, B_z, ldb);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:72: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::chemm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::chemm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasChemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasChemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:73: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zhemm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zhemm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZhemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZhemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:74: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cherk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, *(&alpha_s), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, *(&beta_s), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cherk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, *(&alpha_s), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, *(&beta_s), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &alpha_s, A_c, lda, &beta_s, C_c, ldc);
  cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &alpha_s, A_c, lda, &beta_s, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:75: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zherk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, *(alpha_d), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, *(&beta_d), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zherk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, *(alpha_d), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, *(&beta_d), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_d, A_z, lda, &beta_d, C_z, ldc);
  cublasZherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_d, A_z, lda, &beta_d, C_z, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:76: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::cher2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_s), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_c_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cher2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), A_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_s), C_c_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, &beta_s, C_c, ldc);
  cublasCher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, &beta_s, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:77: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::zher2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_d), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zher2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), A_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_d), C_z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, &beta_d, C_z, ldc);
  cublasZher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, &beta_d, C_z, ldc);
}