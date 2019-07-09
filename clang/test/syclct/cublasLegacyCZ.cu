// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/cublasLegacyCZ.sycl.cpp --match-full-lines %s

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
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::icamax(syclct::get_default_queue(), n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: res = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  int res = cublasIcamax(n, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::izamax(syclct::get_default_queue(), n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIzamax(n, x_Z, incx);

  //cublasI<t>amin
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::icamin(syclct::get_default_queue(), n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIcamin(n, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::izamin(syclct::get_default_queue(), n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result = cublasIzamin(n, x_Z, incx);

  //cublas<t>asum
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::scasum(syclct::get_default_queue(), n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_S = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasScasum(n, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::dzasum(syclct::get_default_queue(), n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_D = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDzasum(n, x_Z, incx);

  //cublas<t>dot
  // CHECK: cl::sycl::float2 resCuComplex;
  // CHECK-NEXT: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::cdotu(syclct::get_default_queue(), n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_temp_buffer);
  // CHECK-NEXT: resCuComplex = cl::sycl::float2(result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].real(), result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].imag());
  // CHECK-NEXT: }
  cuComplex resCuComplex = cublasCdotu(n, x_C, incx, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::cdotc(syclct::get_default_queue(), n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_temp_buffer);
  // CHECK-NEXT: *result_C = cl::sycl::float2(result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].real(), result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].imag());;
  // CHECK-NEXT: }
  *result_C = cublasCdotc(n, x_C, incx, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::zdotu(syclct::get_default_queue(), n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_temp_buffer);
  // CHECK-NEXT: *result_Z = cl::sycl::double2(result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].real(), result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].imag());;
  // CHECK-NEXT: }
  *result_Z = cublasZdotu(n, x_Z, incx, y_Z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::zdotc(syclct::get_default_queue(), n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_temp_buffer);
  // CHECK-NEXT: *result_Z = cl::sycl::double2(result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].real(), result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0].imag());;
  // CHECK-NEXT: }
  *result_Z = cublasZdotc(n, x_Z, incx, y_Z, incy);

  //cublas<t>nrm2
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: cl::sycl::buffer<float,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::scnrm2(syclct::get_default_queue(), n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_S = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_S = cublasScnrm2(n, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: cl::sycl::buffer<double,1> result_temp_buffer(cl::sycl::range<1>(1));
  // CHECK-NEXT: mkl::dznrm2(syclct::get_default_queue(), n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_temp_buffer);
  // CHECK-NEXT: *result_D = result_temp_buffer.get_access<cl::sycl::access::mode::read>()[0];
  // CHECK-NEXT: }
  *result_D = cublasDznrm2(n, x_Z, incx);




  //cublas<t>axpy
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::caxpy(syclct::get_default_queue(), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasCaxpy(n, alpha_C, x_C, incx, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zaxpy(syclct::get_default_queue(), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZaxpy(n, alpha_Z, x_Z, incx, result_Z, incy);

  //cublas<t>copy
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ccopy(syclct::get_default_queue(), n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasCcopy(n, x_C, incx, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zcopy(syclct::get_default_queue(), n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZcopy(n, x_Z, incx, result_Z, incy);


  //cublas<t>rot
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::csrot(syclct::get_default_queue(), n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, *x_S, *y_S);
  // CHECK-NEXT: }
  cublasCsrot(n, x_C, incx, y_C, incy, *x_S, *y_S);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zdrot(syclct::get_default_queue(), n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, *x_D, *y_D);
  // CHECK-NEXT: }
  cublasZdrot(n, x_Z, incx, y_Z, incy, *x_D, *y_D);


  //cublas<t>scal
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cscal(syclct::get_default_queue(), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  cublasCscal(n, alpha_C, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zscal(syclct::get_default_queue(), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  cublasZscal(n, alpha_Z, x_Z, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::csscal(syclct::get_default_queue(), n, alpha_S, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  cublasCsscal(n, alpha_S, x_C, incx);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zdscal(syclct::get_default_queue(), n, alpha_D, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx);
  // CHECK-NEXT: }
  cublasZdscal(n, alpha_D, x_Z, incx);

  //cublas<t>swap
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cswap(syclct::get_default_queue(), n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasCswap(n, x_C, incx, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zswap(syclct::get_default_queue(), n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZswap(n, x_Z, incx, y_Z, incy);

  //level2
  //cublas<t>gbmv
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cgbmv(syclct::get_default_queue(), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_C).x(),(beta_C).y()), result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasCgbmv('N', m, n, m, n, alpha_C, x_C, lda, y_C, incx, beta_C, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zgbmv(syclct::get_default_queue(), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_Z).x(),(beta_Z).y()), result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZgbmv( 'N', m, n, m, n, alpha_Z, x_Z, lda, y_Z, incx, beta_Z, result_Z, incy);

  //cublas<t>gemv
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cgemv(syclct::get_default_queue(), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_C).x(),(beta_C).y()), result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasCgemv('N', m, n, alpha_C, x_C, lda, y_C, incx, beta_C, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zgemv(syclct::get_default_queue(), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_Z).x(),(beta_Z).y()), result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZgemv('N', m, n, alpha_Z, x_Z, lda, y_Z, incx, beta_Z, result_Z, incy);

  //cublas<t>ger
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cgeru(syclct::get_default_queue(), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasCgeru(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cgerc(syclct::get_default_queue(), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasCgerc(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zgeru(syclct::get_default_queue(), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasZgeru(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zgerc(syclct::get_default_queue(), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasZgerc(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);



  //cublas<t>tbmv
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctbmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasCtbmv('U', 'N', 'U', n, n, x_C, lda, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztbmv(syclct::get_default_queue(), ((('u')=='L'||('u')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('u')=='N'||('u')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZtbmv('u', 'N', 'u', n, n, x_Z, lda, result_Z, incy);

  //cublas<t>tbsv
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctbsv(syclct::get_default_queue(), ((('L')=='L'||('L')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasCtbsv('L', 'N', 'U', n, n, x_C, lda, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztbsv(syclct::get_default_queue(), ((('l')=='L'||('l')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZtbsv('l', 'N', 'U', n, n, x_Z, lda, result_Z, incy);

  //cublas<t>tpmv
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctpmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasCtpmv('U', 'N', 'U', n, x_C, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztpmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZtpmv('U', 'N', 'U', n, x_Z, result_Z, incy);

  //cublas<t>tpsv
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctpsv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasCtpsv('U', 'N', 'U', n, x_C, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztpsv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZtpsv('U', 'N', 'U', n, x_Z, result_Z, incy);

  //cublas<t>trmv
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctrmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasCtrmv('U', 'N', 'U', n, x_C, lda, result_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztrmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZtrmv('U', 'N', 'U', n, x_Z, lda, result_Z, incy);

  //cublas<t>trsv
  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(result_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctrsv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasCtrsv('U', 'N', 'U', n, x_C, lda, result_C, incy);


  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(result_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(result_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztrsv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('U')=='N'||('U')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), n, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, result_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZtrsv('U', 'N', 'U', n, x_Z, lda, result_Z, incy);

  //chemv
  // CHECK: {
  // CHECK-NEXT: auto A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::chemv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_C).x(),(beta_C).y()), y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasChemv ('U', n, alpha_C, A_C, lda, x_C, incx, beta_C, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zhemv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_Z).x(),(beta_Z).y()), y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZhemv ('U', n, alpha_Z, A_Z, lda, x_Z, incx, beta_Z, y_Z, incy);

  // CHECK: {
  // CHECK-NEXT: auto A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::chbmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, k, std::complex<float>((alpha_C).x(),(alpha_C).y()), A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_C).x(),(beta_C).y()), y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasChbmv ('U', n, k, alpha_C, A_C, lda, x_C, incx, beta_C, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zhbmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, k, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_Z).x(),(beta_Z).y()), y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZhbmv ('U', n, k, alpha_Z, A_Z, lda, x_Z, incx, beta_Z, y_Z, incy);

  // CHECK: {
  // CHECK-NEXT: auto A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::chpmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<float>((beta_C).x(),(beta_C).y()), y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasChpmv('U', n, alpha_C, A_C, x_C, incx, beta_C, y_C, incy);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zhpmv(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, std::complex<double>((beta_Z).x(),(beta_Z).y()), y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy);
  // CHECK-NEXT: }
  cublasZhpmv('U', n, alpha_Z, A_Z, x_Z, incx, beta_Z, y_Z, incy);

  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cher(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasCher ('U', n, alpha_S, x_C, incx, C_C, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zher(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasZher ('U', n, alpha_D, x_Z, incx, C_Z, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cher2(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasCher2 ('U', n, alpha_C, x_C, incx, y_C, incy, C_C, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zher2(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda);
  // CHECK-NEXT: }
  cublasZher2 ('U', n, alpha_Z, x_Z, incx, y_Z, incy, C_Z, lda);

  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::chpr(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_S, x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasChpr ('U', n, alpha_S, x_C, incx, C_C);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zhpr(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, alpha_D, x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasZhpr ('U', n, alpha_D, x_Z, incx, C_Z);

  // CHECK: {
  // CHECK-NEXT: auto x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(x_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(y_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::chpr2(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<float>((alpha_C).x(),(alpha_C).y()), x_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasChpr2 ('U', n, alpha_C, x_C, incx, y_C, incy, C_C);

  // CHECK: {
  // CHECK-NEXT: auto x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(x_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(x_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(y_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(y_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zhpr2(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), x_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incx, y_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, incy, C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}});
  // CHECK-NEXT: }
  cublasZhpr2 ('U', n, alpha_Z, x_Z, incx, y_Z, incy, C_Z);


  //level3
  // CHECK: {
  // CHECK-NEXT: auto A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cgemm(syclct::get_default_queue(), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, k, std::complex<float>((alpha_C).x(),(alpha_C).y()), A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<float>((beta_C).x(),(beta_C).y()), C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasCgemm('N', 'N', m, n, k, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zgemm(syclct::get_default_queue(), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), m, n, k, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<double>((beta_Z).x(),(beta_Z).y()), C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasZgemm('N', 'N', m, n, k, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // cublas<T>symm
  // CHECK: {
  // CHECK-NEXT: auto A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::csymm(syclct::get_default_queue(), ((('R')=='L'||('R')=='l')?(mkl::side::left):(mkl::side::right)), ((('L')=='L'||('L')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<float>((beta_C).x(),(beta_C).y()), C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasCsymm('R', 'L', m, n, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zsymm(syclct::get_default_queue(), ((('r')=='L'||('r')=='l')?(mkl::side::left):(mkl::side::right)), ((('L')=='L'||('L')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<double>((beta_Z).x(),(beta_Z).y()), C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasZsymm('r', 'L', m, n, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::csyrk(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('T')=='N'||('T')=='n')?(mkl::transpose::nontrans):((('T')=='T'||('T')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, std::complex<float>((alpha_C).x(),(alpha_C).y()), A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, std::complex<float>((beta_C).x(),(beta_C).y()), C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasCsyrk('U', 'T', n, k, alpha_C, A_C, lda, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zsyrk(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('t')=='N'||('t')=='n')?(mkl::transpose::nontrans):((('t')=='T'||('t')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, std::complex<double>((beta_Z).x(),(beta_Z).y()), C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasZsyrk('U', 't', n, k, alpha_Z, A_Z, lda, beta_Z, C_Z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cherk(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('t')=='N'||('t')=='n')?(mkl::transpose::nontrans):((('t')=='T'||('t')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, alpha_S, A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, beta_S, C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasCherk('U', 't', n, k, alpha_S, A_C, lda, beta_S, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zherk(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('t')=='N'||('t')=='n')?(mkl::transpose::nontrans):((('t')=='T'||('t')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, alpha_D, A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, beta_D, C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasZherk('U', 't', n, k, alpha_D, A_Z, lda, beta_D, C_Z, ldc);

  // cublas<T>syr2k
  // CHECK: {
  // CHECK-NEXT: auto A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::csyr2k(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('C')=='N'||('C')=='n')?(mkl::transpose::nontrans):((('C')=='T'||('C')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, std::complex<float>((alpha_C).x(),(alpha_C).y()), A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<float>((beta_C).x(),(beta_C).y()), C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasCsyr2k('U', 'C', n, k, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zsyr2k(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('c')=='N'||('c')=='n')?(mkl::transpose::nontrans):((('c')=='T'||('c')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<double>((beta_Z).x(),(beta_Z).y()), C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasZsyr2k('U', 'c', n, k, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::cher2k(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('c')=='N'||('c')=='n')?(mkl::transpose::nontrans):((('c')=='T'||('c')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, std::complex<float>((alpha_C).x(),(alpha_C).y()), A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, beta_S, C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasCher2k('U', 'c', n, k, alpha_C, A_C, lda, B_C, ldb, beta_S, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zher2k(syclct::get_default_queue(), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('c')=='N'||('c')=='n')?(mkl::transpose::nontrans):((('c')=='T'||('c')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), n, k, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, beta_D, C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasZher2k('U', 'c', n, k, alpha_Z, A_Z, lda, B_Z, ldb, beta_D, C_Z, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> B_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::chemm(syclct::get_default_queue(), ((('R')=='L'||('R')=='l')?(mkl::side::left):(mkl::side::right)), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<float>((beta_C).x(),(beta_C).y()), C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasChemm ('R', 'U', m, n, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> B_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::zhemm(syclct::get_default_queue(), ((('R')=='L'||('R')=='l')?(mkl::side::left):(mkl::side::right)), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, std::complex<double>((beta_Z).x(),(beta_Z).y()), C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasZhemm ('R', 'U', m, n, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // cublas<T>trsm
  // CHECK: {
  // CHECK-NEXT: auto A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_C);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(C_C_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: mkl::ctrsm(syclct::get_default_queue(), ((('L')=='L'||('L')=='l')?(mkl::side::left):(mkl::side::right)), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('n')=='N'||('n')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), A_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, C_C_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasCtrsm('L', 'U', 'N', 'n', m, n, alpha_C, A_C, lda, C_C, ldc);

  // CHECK: {
  // CHECK-NEXT: auto A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_Z);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(C_Z_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: mkl::ztrsm(syclct::get_default_queue(), ((('l')=='L'||('l')=='l')?(mkl::side::left):(mkl::side::right)), ((('U')=='L'||('U')=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), ((('N')=='N'||('N')=='n')?(mkl::transpose::nontrans):((('N')=='T'||('N')=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), ((('N')=='N'||('N')=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), A_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, C_Z_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  // CHECK-NEXT: }
  cublasZtrsm('l', 'U', 'N', 'N', m, n, alpha_Z, A_Z, lda, C_Z, ldc);


}
