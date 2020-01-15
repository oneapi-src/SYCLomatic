// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasLegacyCZ.dp.cpp --match-full-lines %s

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
  const cuComplex *A_C = 0;
  const cuComplex *B_C = 0;
  cuComplex *C_C = 0;
  cuComplex alpha_C = make_cuComplex(1,0);
  cuComplex beta_C = make_cuComplex(0,0);
  const cuDoubleComplex *A_Z = 0;
  const cuDoubleComplex *B_Z = 0;
  cuDoubleComplex *C_Z = 0;
  cuDoubleComplex alpha_Z = make_cuDoubleComplex(1,0);
  cuDoubleComplex beta_Z = make_cuDoubleComplex(0,0);

  cuComplex *x_C = 0;
  cuDoubleComplex *x_Z = 0;
  cuComplex *y_C = 0;
  cuDoubleComplex *y_Z = 0;
  int incx = 1;
  int incy = 1;
  int *result = 0;
  cuComplex *result_C = 0;
  cuDoubleComplex *result_Z = 0;
  float *result_S = 0;
  double *result_D = 0;

  float *x_f = 0;
  float *y_f = 0;
  double *x_d = 0;
  double *y_d = 0;
  float *x_S = 0;
  float *y_S = 0;
  double *x_D = 0;
  double *y_D = 0;

  float alpha_S = 0;
  double alpha_D = 0;
  float beta_S = 0;
  double beta_D = 0;
  //level1

  //cublasI<t>amax
  // CHECK: int res;
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamax(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: res = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  int res = cublasIcamax(n, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamax(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIzamax(n, x_Z, incx);

  //cublasI<t>amin
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamin(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIcamin(n, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::iamin(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIzamin(n, x_Z, incx);

  //cublas<t>asum
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: cl::sycl::buffer<float> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::asum(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_S = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasScasum(n, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: cl::sycl::buffer<double> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::asum(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_D = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDzasum(n, x_Z, incx);

  //cublas<t>dot
  // CHECK: cl::sycl::float2 resCuComplex;
  // CHECK-NEXT: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::dotu(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, result_temp_buffer);
  // CHECK-NEXT: resCuComplex = cl::sycl::float2(result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].real(), result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].imag());
  // CHECK-NEXT: }
  cuComplex resCuComplex = cublasCdotu(n, x_C, incx, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::dotc(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, result_temp_buffer);
  // CHECK-NEXT: *result_C = cl::sycl::float2(result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].real(), result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].imag());;
  // CHECK-NEXT: }
  *result_C = cublasCdotc(n, x_C, incx, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::dotu(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, result_temp_buffer);
  // CHECK-NEXT: *result_Z = cl::sycl::double2(result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].real(), result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].imag());;
  // CHECK-NEXT: }
  *result_Z = cublasZdotu(n, x_Z, incx, y_Z, incy);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::dotc(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, result_temp_buffer);
  // CHECK-NEXT: *result_Z = cl::sycl::double2(result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].real(), result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].imag());;
  // CHECK-NEXT: }
  *result_Z = cublasZdotc(n, x_Z, incx, y_Z, incy);

  //cublas<t>nrm2
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: cl::sycl::buffer<float> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::nrm2(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_S = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasScnrm2(n, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: cl::sycl::buffer<double> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::blas::nrm2(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_D = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDznrm2(n, x_Z, incx);




  //cublas<t>axpy
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::axpy(dpct::get_default_queue(), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCaxpy(n, alpha_C, x_C, incx, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::axpy(dpct::get_default_queue(), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZaxpy(n, alpha_Z, x_Z, incx, result_Z, incy);

  //cublas<t>copy
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::copy(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCcopy(n, x_C, incx, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::copy(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZcopy(n, x_Z, incx, result_Z, incy);


  //cublas<t>rot
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::rot(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, *x_S, *y_S);
  // CHECK-NEXT: }
  cublasCsrot(n, x_C, incx, y_C, incy, *x_S, *y_S);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::rot(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, *x_D, *y_D);
  // CHECK-NEXT: }
  cublasZdrot(n, x_Z, incx, y_Z, incy, *x_D, *y_D);


  //cublas<t>scal
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::scal(dpct::get_default_queue(), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  cublasCscal(n, alpha_C, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::scal(dpct::get_default_queue(), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  cublasZscal(n, alpha_Z, x_Z, incx);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::scal(dpct::get_default_queue(), n, alpha_S, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  cublasCsscal(n, alpha_S, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::scal(dpct::get_default_queue(), n, alpha_D, buffer_ct{{[0-9]+}}, incx);
  // CHECK-NEXT: }
  cublasZdscal(n, alpha_D, x_Z, incx);

  //cublas<t>swap
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::swap(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCswap(n, x_C, incx, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::swap(dpct::get_default_queue(), n, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZswap(n, x_Z, incx, y_Z, incy);

  //level2
  //cublas<t>gbmv
  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::gbmv(dpct::get_default_queue(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_C).x(),(beta_C).y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCgbmv('N', m, n, m, n, alpha_C, x_C, lda, y_C, incx, beta_C, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::gbmv(dpct::get_default_queue(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_Z).x(),(beta_Z).y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZgbmv( 'N', m, n, m, n, alpha_Z, x_Z, lda, y_Z, incx, beta_Z, result_Z, incy);

  //cublas<t>gemv
  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::gemv(dpct::get_default_queue(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_C).x(),(beta_C).y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCgemv('N', m, n, alpha_C, x_C, lda, y_C, incx, beta_C, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::gemv(dpct::get_default_queue(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_Z).x(),(beta_Z).y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZgemv('N', m, n, alpha_Z, x_Z, lda, y_Z, incx, beta_Z, result_Z, incy);

  //cublas<t>ger
  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::geru(dpct::get_default_queue(), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasCgeru(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::gerc(dpct::get_default_queue(), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasCgerc(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::geru(dpct::get_default_queue(), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasZgeru(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);

  // CHECK: {
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::gerc(dpct::get_default_queue(), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasZgerc(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);



  //cublas<t>tbmv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::tbmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCtbmv('U', 'N', 'U', n, n, x_C, lda, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'u';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'u';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::tbmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZtbmv('u', 'N', 'u', n, n, x_Z, lda, result_Z, incy);

  //cublas<t>tbsv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'L';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::tbsv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCtbsv('L', 'N', 'U', n, n, x_C, lda, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'l';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::tbsv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZtbsv('l', 'N', 'U', n, n, x_Z, lda, result_Z, incy);

  //cublas<t>tpmv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::tpmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCtpmv('U', 'N', 'U', n, x_C, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::tpmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZtpmv('U', 'N', 'U', n, x_Z, result_Z, incy);

  //cublas<t>tpsv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::tpsv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCtpsv('U', 'N', 'U', n, x_C, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::tpsv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZtpsv('U', 'N', 'U', n, x_Z, result_Z, incy);

  //cublas<t>trmv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::trmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCtrmv('U', 'N', 'U', n, x_C, lda, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::trmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZtrmv('U', 'N', 'U', n, x_Z, lda, result_Z, incy);

  //cublas<t>trsv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::trsv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasCtrsv('U', 'N', 'U', n, x_C, lda, result_C, incy);


  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto diagtype_ct2 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::trsv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct2)=='N'||(diagtype_ct2)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZtrsv('U', 'N', 'U', n, x_Z, lda, result_Z, incy);

  //chemv
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::hemv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_C).x(),(beta_C).y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasChemv ('U', n, alpha_C, A_C, lda, x_C, incx, beta_C, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::hemv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_Z).x(),(beta_Z).y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZhemv ('U', n, alpha_Z, A_Z, lda, x_Z, incx, beta_Z, y_Z, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::hbmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, k, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_C).x(),(beta_C).y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasChbmv ('U', n, k, alpha_C, A_C, lda, x_C, incx, beta_C, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::hbmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, k, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_Z).x(),(beta_Z).y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZhbmv ('U', n, k, alpha_Z, A_Z, lda, x_Z, incx, beta_Z, y_Z, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::hpmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx, std::complex<float>((beta_C).x(),(beta_C).y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasChpmv('U', n, alpha_C, A_C, x_C, incx, beta_C, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::hpmv(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, buffer_ct{{[0-9]+}}, incx, std::complex<double>((beta_Z).x(),(beta_Z).y()), buffer_ct{{[0-9]+}}, incy);
  // CHECK-NEXT: }
  cublasZhpmv('U', n, alpha_Z, A_Z, x_Z, incx, beta_Z, y_Z, incy);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::her(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasCher ('U', n, alpha_S, x_C, incx, C_C, lda);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::her(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasZher ('U', n, alpha_D, x_Z, incx, C_Z, lda);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::her2(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasCher2 ('U', n, alpha_C, x_C, incx, y_C, incy, C_C, lda);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::her2(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}}, lda);
  // CHECK-NEXT: }
  cublasZher2 ('U', n, alpha_Z, x_Z, incx, y_Z, incy, C_Z, lda);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::hpr(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasChpr ('U', n, alpha_S, x_C, incx, C_C);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::hpr(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasZhpr ('U', n, alpha_D, x_Z, incx, C_Z);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::hpr2(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasChpr2 ('U', n, alpha_C, x_C, incx, y_C, incy, C_C);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::hpr2(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, incx, buffer_ct{{[0-9]+}}, incy, buffer_ct{{[0-9]+}});
  // CHECK-NEXT: }
  cublasZhpr2 ('U', n, alpha_Z, x_Z, incx, y_Z, incy, C_Z);


  //level3
  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::gemm(dpct::get_default_queue(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, k, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<float>((beta_C).x(),(beta_C).y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCgemm('N', 'N', m, n, k, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto transpose_ct0 = 'N';
  // CHECK-NEXT: auto transpose_ct1 = 'N';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::gemm(dpct::get_default_queue(), (((transpose_ct0)=='N'||(transpose_ct0)=='n')?(mkl::transpose::nontrans):(((transpose_ct0)=='T'||(transpose_ct0)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, k, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<double>((beta_Z).x(),(beta_Z).y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZgemm('N', 'N', m, n, k, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // cublas<T>symm
  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'R';
  // CHECK-NEXT: auto fillmode_ct1 = 'L';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::symm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<float>((beta_C).x(),(beta_C).y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCsymm('R', 'L', m, n, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'r';
  // CHECK-NEXT: auto fillmode_ct1 = 'L';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::symm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<double>((beta_Z).x(),(beta_Z).y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZsymm('r', 'L', m, n, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'T';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::syrk(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, lda, std::complex<float>((beta_C).x(),(beta_C).y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCsyrk('U', 'T', n, k, alpha_C, A_C, lda, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 't';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::syrk(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, lda, std::complex<double>((beta_Z).x(),(beta_Z).y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZsyrk('U', 't', n, k, alpha_Z, A_Z, lda, beta_Z, C_Z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 't';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::herk(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, alpha_S, buffer_ct{{[0-9]+}}, lda, beta_S, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCherk('U', 't', n, k, alpha_S, A_C, lda, beta_S, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 't';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::herk(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, alpha_D, buffer_ct{{[0-9]+}}, lda, beta_D, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZherk('U', 't', n, k, alpha_D, A_Z, lda, beta_D, C_Z, ldc);

  // cublas<T>syr2k
  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'C';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::syr2k(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<float>((beta_C).x(),(beta_C).y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCsyr2k('U', 'C', n, k, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'c';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::syr2k(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<double>((beta_Z).x(),(beta_Z).y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZsyr2k('U', 'c', n, k, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'c';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::her2k(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, beta_S, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCher2k('U', 'c', n, k, alpha_C, A_C, lda, B_C, ldb, beta_S, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto fillmode_ct0 = 'U';
  // CHECK-NEXT: auto transpose_ct1 = 'c';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::her2k(dpct::get_default_queue(), (((fillmode_ct0)=='L'||(fillmode_ct0)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct1)=='N'||(transpose_ct1)=='n')?(mkl::transpose::nontrans):(((transpose_ct1)=='T'||(transpose_ct1)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, beta_D, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZher2k('U', 'c', n, k, alpha_Z, A_Z, lda, B_Z, ldb, beta_D, C_Z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'R';
  // CHECK-NEXT: auto fillmode_ct1 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::hemm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<float>((beta_C).x(),(beta_C).y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasChemm ('R', 'U', m, n, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'R';
  // CHECK-NEXT: auto fillmode_ct1 = 'U';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(B_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::hemm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldb, std::complex<double>((beta_Z).x(),(beta_Z).y()), buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZhemm ('R', 'U', m, n, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // cublas<T>trsm
  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'L';
  // CHECK-NEXT: auto fillmode_ct1 = 'U';
  // CHECK-NEXT: auto transpose_ct2 = 'N';
  // CHECK-NEXT: auto diagtype_ct3 = 'n';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::blas::trsm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct2)=='N'||(transpose_ct2)=='n')?(mkl::transpose::nontrans):(((transpose_ct2)=='T'||(transpose_ct2)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct3)=='N'||(diagtype_ct3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasCtrsm('L', 'U', 'N', 'n', m, n, alpha_C, A_C, lda, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'l';
  // CHECK-NEXT: auto fillmode_ct1 = 'U';
  // CHECK-NEXT: auto transpose_ct2 = 'N';
  // CHECK-NEXT: auto diagtype_ct3 = 'N';
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::blas::trsm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct2)=='N'||(transpose_ct2)=='n')?(mkl::transpose::nontrans):(((transpose_ct2)=='T'||(transpose_ct2)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct3)=='N'||(diagtype_ct3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), buffer_ct{{[0-9]+}}, lda, buffer_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZtrsm('l', 'U', 'N', 'N', m, n, alpha_Z, A_Z, lda, C_Z, ldc);


}
