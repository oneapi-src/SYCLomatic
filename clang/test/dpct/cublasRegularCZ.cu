// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
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
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: status = (mkl::blas::iamax(handle, n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer), 0);
  // CHECK-NEXT: buffer_ct{{[0-9]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamax(handle, n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: buffer_ct{{[0-9]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIcamax(handle, n, x_c, incx, result);
  cublasIcamax(handle, n, x_c, incx, result);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: status = (mkl::blas::iamax(handle, n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer), 0);
  // CHECK-NEXT: buffer_ct{{[0-9]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamax(handle, n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: buffer_ct{{[0-9]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIzamax(handle, n, x_z, incx, result);
  cublasIzamax(handle, n, x_z, incx, result);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: status = (mkl::blas::iamin(handle, n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer), 0);
  // CHECK-NEXT: buffer_ct{{[0-9]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamin(handle, n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: buffer_ct{{[0-9]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIcamin(handle, n, x_c, incx, result);
  cublasIcamin(handle, n, x_c, incx, result);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: status = (mkl::blas::iamin(handle, n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer), 0);
  // CHECK-NEXT: buffer_ct{{[0-9]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result);
  // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(int)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamin(handle, n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: buffer_ct{{[0-9]+}}.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  status = cublasIzamin(handle, n, x_z, incx, result);
  cublasIzamin(handle, n, x_z, incx, result);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_f);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::blas::asum(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_f);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::asum(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasScasum(handle, n, x_c, incx, result_f);
  cublasScasum(handle, n, x_c, incx, result_f);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_d);
  // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::blas::asum(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_d);
  // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::asum(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}});
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
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::axpy(handle, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::axpy(handle, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCaxpy(handle, n, alpha_c, x_c, incx, y_c, incy);
  cublasCaxpy(handle, n, alpha_c, x_c, incx, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::axpy(handle, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::axpy(handle, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZaxpy(handle, n, alpha_z, x_z, incx, y_z, incy);
  cublasZaxpy(handle, n, alpha_z, x_z, incx, y_z, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::copy(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::copy(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCcopy(handle, n, x_c, incx, y_c, incy);
  cublasCcopy(handle, n, x_c, incx, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::copy(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::copy(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZcopy(handle, n, x_z, incx, y_z, incy);
  cublasZcopy(handle, n, x_z, incx, y_z, incy);

  cuComplex* result_c = 0;
  cuDoubleComplex* result_z = 0;

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::dotu(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::dotu(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasCdotu(handle, n, x_c, incx, y_c, incy, result_c);
  cublasCdotu(handle, n, x_c, incx, y_c, incy, result_c);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::dotc(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::dotc(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasCdotc(handle, n, x_c, incx, y_c, incy, result_c);
  cublasCdotc(handle, n, x_c, incx, y_c, incy, result_c);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::dotu(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::dotu(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasZdotu(handle, n, x_z, incx, y_z, incy, result_z);
  cublasZdotu(handle, n, x_z, incx, y_z, incy, result_z);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::dotc(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::dotc(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasZdotc(handle, n, x_z, incx, y_z, incy, result_z);
  cublasZdotc(handle, n, x_z, incx, y_z, incy, result_z);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_f);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::blas::nrm2(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_f);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::nrm2(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasScnrm2(handle, n, x_c, incx, result_f);
  cublasScnrm2(handle, n, x_c, incx, result_f);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_d);
  // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: status = (mkl::blas::nrm2(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_d);
  // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: mkl::blas::nrm2(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}});
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
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::rot(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, *(c_f), *(s_f)), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::rot(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, *(c_f), *(s_f));
  // CHECK-NEXT: }
  status = cublasCsrot(handle, n, x_c, incx, y_c, incy, c_f, s_f);
  cublasCsrot(handle, n, x_c, incx, y_c, incy, c_f, s_f);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::rot(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, *(c_d), *(s_d)), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::rot(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, *(c_d), *(s_d));
  // CHECK-NEXT: }
  status = cublasZdrot(handle, n, x_z, incx, y_z, incy, c_d, s_d);
  cublasZdrot(handle, n, x_z, incx, y_z, incy, c_d, s_d);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(c_f);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(s_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::rotg(handle, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(c_f);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(s_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::rotg(handle, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasCrotg(handle, x_c, y_c, c_f, s_c);
  cublasCrotg(handle, x_c, y_c, c_f, s_c);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(c_d);
  // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(s_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::rotg(handle, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(c_d);
  // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(s_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::rotg(handle, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasZrotg(handle, x_z, y_z, c_d, s_z);
  cublasZrotg(handle, x_z, y_z, c_d, s_z);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::scal(handle, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::scal(handle, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCscal(handle, n, alpha_c, x_c, incx);
  cublasCscal(handle, n, alpha_c, x_c, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::scal(handle, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::scal(handle, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZscal(handle, n, alpha_z, x_z, incx);
  cublasZscal(handle, n, alpha_z, x_z, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::scal(handle, n, *(alpha_f), buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::scal(handle, n, *(alpha_f), buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCsscal(handle, n, alpha_f, x_c, incx);
  cublasCsscal(handle, n, alpha_f, x_c, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::scal(handle, n, *(alpha_d), buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::scal(handle, n, *(alpha_d), buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZdscal(handle, n, alpha_d, x_z, incx);
  cublasZdscal(handle, n, alpha_d, x_z, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::swap(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::swap(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCswap(handle, n, x_c, incx, y_c, incy);
  cublasCswap(handle, n, x_c, incx, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::swap(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::swap(handle, n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
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
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct1 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::gbmv(handle, (((int)transpose_ct1)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct1)), m, n, kl, ku, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::gbmv(handle, mkl::transpose::nontrans, m, n, kl, ku, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCgbmv(handle, (cublasOperation_t)trans0, m, n, kl, ku, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasCgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct1 = trans1;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::gbmv(handle, (((int)transpose_ct1)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct1)), m, n, kl, ku, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::gbmv(handle, mkl::transpose::nontrans, m, n, kl, ku, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZgbmv(handle, (cublasOperation_t)trans1, m, n, kl, ku, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct1 = trans2;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::gemv(handle, (((int)transpose_ct1)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct1)), m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::gemv(handle, mkl::transpose::nontrans, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasCgemv(handle, (cublasOperation_t)trans2, m, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasCgemv(handle, CUBLAS_OP_N, m, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct1 = 0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::gemv(handle, (((int)transpose_ct1)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct1)), m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::gemv(handle, mkl::transpose::nontrans, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZgemv(handle, (cublasOperation_t)0, m, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZgemv(handle, CUBLAS_OP_N, m, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::geru(handle, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::geru(handle, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCgeru(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCgeru(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::gerc(handle, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::gerc(handle, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCgerc(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCgerc(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::geru(handle, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::geru(handle, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZgeru(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZgeru(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::gerc(handle, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::gerc(handle, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZgerc(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZgerc(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  int k = 1;
  int fill0 = 0;
  int fill1 = 1;
  int diag0 = 0;
  int diag1 = 1;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = 1;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::tbmv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), (mkl::diag)diag0, n, k, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::tbmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, k, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtbmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)1, (cublasDiagType_t)diag0, n, k, x_c, lda, result_c, incx);
  cublasCtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_c, lda, result_c, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = 2;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::tbmv(handle, (((int)fill1)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), (mkl::diag)diag1, n, k, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::tbmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, k, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtbmv(handle, (cublasFillMode_t)fill1, (cublasOperation_t)2, (cublasDiagType_t)diag1, n, k, x_z, lda, result_z, incx);
  cublasZtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_z, lda, result_z, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::tbsv(handle, (((int)0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), (mkl::diag)0,  n, k, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::tbsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit,  n, k, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtbsv(handle, (cublasFillMode_t)0, (cublasOperation_t)trans0, (cublasDiagType_t)0,  n, k, x_c, lda, result_c, incx);
  cublasCtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_c, lda, result_c, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::tbsv(handle, (((int)1)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), (mkl::diag)1,  n, k, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::tbsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit,  n, k, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtbsv(handle, (cublasFillMode_t)1, (cublasOperation_t)trans0, (cublasDiagType_t)1,  n, k, x_z, lda, result_z, incx);
  cublasZtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_z, lda, result_z, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::tpmv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), (mkl::diag)diag0, n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::tpmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, result_c, incx);
  cublasCtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::tpmv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), (mkl::diag)diag0, n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::tpmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, result_z, incx);
  cublasZtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::tpsv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), (mkl::diag)diag0, n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::tpsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, result_c, incx);
  cublasCtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::tpsv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), (mkl::diag)diag0, n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::tpsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, result_z, incx);
  cublasZtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::trmv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), (mkl::diag)diag0, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::trmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, lda, result_c, incx);
  cublasCtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::trmv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), (mkl::diag)diag0, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::trmv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, lda, result_z, incx);
  cublasZtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::trsv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), (mkl::diag)diag0, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::trsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasCtrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, lda, result_c, incx);
  cublasCtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::trsv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), (mkl::diag)diag0, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::trsv(handle, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  status = cublasZtrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, lda, result_z, incx);
  cublasZtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::hemv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::hemv(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasChemv(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasChemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::hemv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::hemv(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZhemv(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZhemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::hbmv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::hbmv(handle, mkl::uplo::lower, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasChbmv(handle, (cublasFillMode_t)fill0, n, k, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasChbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::hbmv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::hbmv(handle, mkl::uplo::lower, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZhbmv(handle, (cublasFillMode_t)fill0, n, k, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZhbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::hpmv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::hpmv(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasChpmv(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, x_c, incx, beta_c, y_c, incy);
  cublasChpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, x_c, incx, beta_c, y_c, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::hpmv(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, incy), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::hpmv(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  status = cublasZhpmv(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, x_z, incx, beta_z, y_z, incy);
  cublasZhpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, x_z, incx, beta_z, y_z, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::her(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, *(alpha_f), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::her(handle, mkl::uplo::lower, n, *(alpha_f), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCher(handle, (cublasFillMode_t)fill0, n, alpha_f, x_c, incx, result_c, lda);
  cublasCher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c, lda);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::her(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, *(alpha_d), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::her(handle, mkl::uplo::lower, n, *(alpha_d), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZher(handle, (cublasFillMode_t)fill0, n, alpha_d, x_z, incx, result_z, lda);
  cublasZher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z, lda);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::her2(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::her2(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasCher2(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::her2(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::her2(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  status = cublasZher2(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::hpr(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, *(alpha_f), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::hpr(handle, mkl::uplo::lower, n, *(alpha_f), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasChpr(handle, (cublasFillMode_t)fill0, n, alpha_f, x_c, incx, result_c);
  cublasChpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::hpr(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, *(alpha_d), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::hpr(handle, mkl::uplo::lower, n, *(alpha_d), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasZhpr(handle, (cublasFillMode_t)fill0, n, alpha_d, x_z, incx, result_z);
  cublasZhpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::hpr2(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::hpr2(handle, mkl::uplo::lower, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasChpr2(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, y_c, incy, result_c);
  cublasChpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::hpr2(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::hpr2(handle, mkl::uplo::lower, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  status = cublasZhpr2(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, y_z, incy, result_z);
  cublasZhpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z);

  int N = 100;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct1 = trans0;
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::gemm(handle, (((int)transpose_ct1)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct1)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), N, N, N, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, N), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, N);
  // CHECK-NEXT: }
  status = cublasCgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, N, N, N, alpha_c, x_c, N, y_c, N, beta_c, result_c, N);
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha_c, x_c, N, y_c, N, beta_c, result_c, N);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct1 = trans0;
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::gemm(handle, (((int)transpose_ct1)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct1)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), N, N, N, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, N), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(x_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(y_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(result_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, N);
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

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:62: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct1 = trans0;
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::cgemm3m(handle, (((int)transpose_ct1)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct1)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), m, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::cgemm3m(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, m, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCgemm3m(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:63: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct1 = trans0;
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::zgemm3m(handle, (((int)transpose_ct1)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct1)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), m, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::zgemm3m(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, m, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZgemm3m(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  int side0 = 0;
  int side1 = 1;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:64: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::symm(handle, (mkl::side)side0, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::symm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCsymm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:65: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::symm(handle, (mkl::side)side1, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::symm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZsymm(handle, (cublasSideMode_t)side1, (cublasFillMode_t)fill0, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:66: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::syrk(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::syrk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);
  cublasCsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:67: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::syrk(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::syrk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);
  cublasZsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:68: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::syr2k(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::syr2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:69: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::syr2k(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::syr2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:70: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct3 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::trsm(handle, (mkl::side)0, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct3)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct3)), (mkl::diag)diag0, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::trsm(handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb);
  // CHECK-NEXT: }
  status = cublasCtrsm(handle, (cublasSideMode_t)0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, alpha_c, A_c, lda, B_c, ldb);
  cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_c, A_c, lda, B_c, ldb);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:71: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct3 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::trsm(handle, (mkl::side)1, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct3)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct3)), (mkl::diag)diag0, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::trsm(handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb);
  // CHECK-NEXT: }
  status = cublasZtrsm(handle, (cublasSideMode_t)1, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, alpha_z, A_z, lda, B_z, ldb);
  cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_z, A_z, lda, B_z, ldb);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:72: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::hemm(handle, (mkl::side)side0, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::hemm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<float>((beta_c)->x(),(beta_c)->y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasChemm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasChemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:73: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::hemm(handle, (mkl::side)side0, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::hemm(handle, mkl::side::left, mkl::uplo::lower, m, n, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<double>((beta_z)->x(),(beta_z)->y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZhemm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZhemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:74: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::herk(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), n, k, *(&alpha_s), buffer_ct{{[0-9]+}}, lda, *(&beta_s), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::herk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, *(&alpha_s), buffer_ct{{[0-9]+}}, lda, *(&beta_s), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCherk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_s, A_c, lda, &beta_s, C_c, ldc);
  cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &alpha_s, A_c, lda, &beta_s, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:75: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::herk(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), n, k, *(alpha_d), buffer_ct{{[0-9]+}}, lda, *(&beta_d), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::herk(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, *(alpha_d), buffer_ct{{[0-9]+}}, lda, *(&beta_d), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZherk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_d, A_z, lda, &beta_d, C_z, ldc);
  cublasZherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_d, A_z, lda, &beta_d, C_z, ldc);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:76: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: status = (mkl::blas::her2k(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, *(&beta_s), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_c);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::her2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<float>((alpha_c)->x(),(alpha_c)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, *(&beta_s), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCher2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, B_c, ldb, &beta_s, C_c, ldc);
  cublasCher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, &beta_s, C_c, ldc);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:77: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct2 = trans0;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: status = (mkl::blas::her2k(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, *(&beta_d), buffer_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(A_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(B_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(C_z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::her2k(handle, mkl::uplo::lower, mkl::transpose::nontrans, n, k, std::complex<double>((alpha_z)->x(),(alpha_z)->y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, *(&beta_d), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZher2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, B_z, ldb, &beta_d, C_z, ldc);
  cublasZher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, &beta_d, C_z, ldc);
}
