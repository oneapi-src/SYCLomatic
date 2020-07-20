// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-usm-legacy.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas.h>
#include <cuda_runtime.h>

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

const float2 *A_C;
const float2 *B_C;
float2 *C_C;
float2 alpha_C;
float2 beta_C;
const double2 *A_Z;
const double2 *B_Z;
double2 *C_Z;
double2 alpha_Z;
double2 beta_Z;

const float *x_S = 0;
const double *x_D = 0;
const float *y_S = 0;
const double *y_D = 0;
const float2 *x_C;
const float2 *y_C;
const double2 *x_Z;
const double2 *y_Z;

int incx = 1;
int incy = 1;
int *result = 0;
float *result_S = 0;
double *result_D = 0;
float2 *result_C;
double2 *result_Z;

int elemSize = 4;

int main() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: int status = (C_S = (float *)sycl::malloc_device((n)*(elemSize), q_ct1), 0);
  // CHECK-NEXT: C_S = (float *)sycl::malloc_device((n)*(elemSize), q_ct1);
  cublasStatus status = cublasAlloc(n, elemSize, (void **)&C_S);
  cublasAlloc(n, elemSize, (void **)&C_S);

  // level 1

  // CHECK: int res;
  // CHECK-NEXT: int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  // CHECK-NEXT: oneapi::mkl::blas::iamax(*dpct::get_current_device().get_saved_queue(), n, x_S, incx, res_temp_ptr_ct{{[0-9]+}}).wait();
  // CHECK-NEXT: res = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  int res = cublasIsamax(n, x_S, incx);
  // CHECK: int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  // CHECK-NEXT: oneapi::mkl::blas::iamax(*dpct::get_current_device().get_saved_queue(), n, x_D, incx, res_temp_ptr_ct{{[0-9]+}}).wait();
  // CHECK-NEXT: res = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  res = cublasIdamax(n, x_D, incx);
  // CHECK: int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  // CHECK-NEXT: oneapi::mkl::blas::iamax(*dpct::get_current_device().get_saved_queue(), n, (std::complex<float>*)x_C, incx, res_temp_ptr_ct{{[0-9]+}}).wait();
  // CHECK-NEXT: res = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  res = cublasIcamax(n, x_C, incx);
  // CHECK: int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  // CHECK-NEXT: oneapi::mkl::blas::iamax(*dpct::get_current_device().get_saved_queue(), n, (std::complex<double>*)x_Z, incx, res_temp_ptr_ct{{[0-9]+}}).wait();
  // CHECK-NEXT: res = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  res = cublasIzamax(n, x_Z, incx);

  // Because the return value of origin API is the result value, not the status, so keep using lambda here.
  // CHECK: if([&](){
  // CHECK-NEXT: int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  // CHECK-NEXT: oneapi::mkl::blas::iamax(*dpct::get_current_device().get_saved_queue(), n, (std::complex<double>*)x_Z, incx, res_temp_ptr_ct{{[0-9]+}}).wait();
  // CHECK-NEXT: int64_t res_temp_val_ct{{[0-9]+}} = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT: sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  // CHECK-NEXT: return res_temp_val_ct{{[0-9]+}};
  // CHECK-NEXT: }()){}
  if(cublasIzamax(n, x_Z, incx)){}

  // CHECK: if(0!=[&](){
  // CHECK-NEXT: int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  // CHECK-NEXT: oneapi::mkl::blas::iamax(*dpct::get_current_device().get_saved_queue(), n, (std::complex<double>*)x_Z, incx, res_temp_ptr_ct{{[0-9]+}}).wait();
  // CHECK-NEXT: int64_t res_temp_val_ct{{[0-9]+}} = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT: sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  // CHECK-NEXT: return res_temp_val_ct{{[0-9]+}};
  // CHECK-NEXT: }()){}
  if(0!=cublasIzamax(n, x_Z, incx)){}

  // CHECK: for([&](){
  // CHECK-NEXT: std::complex<float>* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<std::complex<float>>(1, dpct::get_default_queue());
  // CHECK-NEXT: oneapi::mkl::blas::dotc(*dpct::get_current_device().get_saved_queue(), n, (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, res_temp_ptr_ct{{[0-9]+}}).wait();
  // CHECK-NEXT: std::complex<float> res_temp_val_ct{{[0-9]+}} = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT: sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  // CHECK-NEXT: return sycl::float2(res_temp_val_ct{{[0-9]+}}.real(), res_temp_val_ct{{[0-9]+}}.imag());
  // CHECK-NEXT: }();;){}
  for(cublasCdotc(n, x_C, incx, y_C, incy);;){}

  //CHECK:oneapi::mkl::blas::rotm(*dpct::get_current_device().get_saved_queue(), n, result_S, n, result_S, n, const_cast<float*>(x_S)).wait();
  cublasSrotm(n, result_S, n, result_S, n, x_S);
  //CHECK:oneapi::mkl::blas::rotm(*dpct::get_current_device().get_saved_queue(), n, result_D, n, result_D, n, const_cast<double*>(x_D)).wait();
  cublasDrotm(n, result_D, n, result_D, n, x_D);

  // CHECK:oneapi::mkl::blas::copy(*dpct::get_current_device().get_saved_queue(), n, x_S, incx, result_S, incy).wait();
  cublasScopy(n, x_S, incx, result_S, incy);
  // CHECK:oneapi::mkl::blas::copy(*dpct::get_current_device().get_saved_queue(), n, x_D, incx, result_D, incy).wait();
  cublasDcopy(n, x_D, incx, result_D, incy);
  // CHECK:oneapi::mkl::blas::copy(*dpct::get_current_device().get_saved_queue(), n, (std::complex<float>*)x_C, incx, (std::complex<float>*)result_C, incy).wait();
  cublasCcopy(n, x_C, incx, result_C, incy);
  // CHECK:oneapi::mkl::blas::copy(*dpct::get_current_device().get_saved_queue(), n, (std::complex<double>*)x_Z, incx, (std::complex<double>*)result_Z, incy).wait();
  cublasZcopy(n, x_Z, incx, result_Z, incy);

  // CHECK:oneapi::mkl::blas::axpy(*dpct::get_current_device().get_saved_queue(), n, alpha_S, x_S, incx, result_S, incy).wait();
  cublasSaxpy(n, alpha_S, x_S, incx, result_S, incy);
  // CHECK:oneapi::mkl::blas::axpy(*dpct::get_current_device().get_saved_queue(), n, alpha_D, x_D, incx, result_D, incy).wait();
  cublasDaxpy(n, alpha_D, x_D, incx, result_D, incy);
  // CHECK:oneapi::mkl::blas::axpy(*dpct::get_current_device().get_saved_queue(), n, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)x_C, incx, (std::complex<float>*)result_C, incy).wait();
  cublasCaxpy(n, alpha_C, x_C, incx, result_C, incy);
  // CHECK:oneapi::mkl::blas::axpy(*dpct::get_current_device().get_saved_queue(), n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)x_Z, incx, (std::complex<double>*)result_Z, incy).wait();
  cublasZaxpy(n, alpha_Z, x_Z, incx, result_Z, incy);

  // CHECK:oneapi::mkl::blas::scal(*dpct::get_current_device().get_saved_queue(), n, alpha_S, result_S, incx).wait();
  cublasSscal(n, alpha_S, result_S, incx);
  // CHECK:oneapi::mkl::blas::scal(*dpct::get_current_device().get_saved_queue(), n, alpha_D, result_D, incx).wait();
  cublasDscal(n, alpha_D, result_D, incx);
  // CHECK:oneapi::mkl::blas::scal(*dpct::get_current_device().get_saved_queue(), n, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)result_C, incx).wait();
  cublasCscal(n, alpha_C, result_C, incx);
  // CHECK:oneapi::mkl::blas::scal(*dpct::get_current_device().get_saved_queue(), n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)result_Z, incx).wait();
  cublasZscal(n, alpha_Z, result_Z, incx);

  // CHECK: float* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, dpct::get_default_queue());
  // CHECK-NEXT: oneapi::mkl::blas::nrm2(*dpct::get_current_device().get_saved_queue(), n, x_S, incx, res_temp_ptr_ct{{[0-9]+}}).wait();
  // CHECK-NEXT: *result_S = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT: sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  *result_S = cublasSnrm2(n, x_S, incx);
  // CHECK: double* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, dpct::get_default_queue());
  // CHECK-NEXT: oneapi::mkl::blas::nrm2(*dpct::get_current_device().get_saved_queue(), n, x_D, incx, res_temp_ptr_ct{{[0-9]+}}).wait();
  // CHECK-NEXT: *result_D = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT: sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  *result_D = cublasDnrm2(n, x_D, incx);
  // CHECK: float* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, dpct::get_default_queue());
  // CHECK-NEXT: oneapi::mkl::blas::nrm2(*dpct::get_current_device().get_saved_queue(), n, (std::complex<float>*)x_C, incx, res_temp_ptr_ct{{[0-9]+}}).wait();
  // CHECK-NEXT: *result_S = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT: sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  *result_S = cublasScnrm2(n, x_C, incx);
  // CHECK: double* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, dpct::get_default_queue());
  // CHECK-NEXT: oneapi::mkl::blas::nrm2(*dpct::get_current_device().get_saved_queue(), n, (std::complex<double>*)x_Z, incx, res_temp_ptr_ct{{[0-9]+}}).wait();
  // CHECK-NEXT: *result_D = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT: sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  *result_D = cublasDznrm2(n, x_Z, incx);

  // CHECK: std::complex<float>* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<std::complex<float>>(1, dpct::get_default_queue());
  // CHECK-NEXT: oneapi::mkl::blas::dotc(*dpct::get_current_device().get_saved_queue(), n, (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, res_temp_ptr_ct{{[0-9]+}}).wait();
  // CHECK-NEXT: *result_C = sycl::float2(res_temp_ptr_ct{{[0-9]+}}->real(), res_temp_ptr_ct{{[0-9]+}}->imag());
  // CHECK-NEXT: sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  *result_C = cublasCdotc(n, x_C, incx, y_C, incy);

  // CHECK: std::complex<double>* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<std::complex<double>>(1, dpct::get_default_queue());
  // CHECK-NEXT: oneapi::mkl::blas::dotu(*dpct::get_current_device().get_saved_queue(), n, (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, res_temp_ptr_ct{{[0-9]+}}).wait();
  // CHECK-NEXT: *result_Z = sycl::double2(res_temp_ptr_ct{{[0-9]+}}->real(), res_temp_ptr_ct{{[0-9]+}}->imag());
  // CHECK-NEXT: sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  *result_Z = cublasZdotu(n, x_Z, incx, y_Z, incy);

  //level 2

  // CHECK:oneapi::mkl::blas::gemv(*dpct::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy).wait();
  cublasSgemv('N', m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);
  // CHECK:oneapi::mkl::blas::gemv(*dpct::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy).wait();
  cublasDgemv('N', m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);
  // CHECK:oneapi::mkl::blas::gemv(*dpct::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, m, n, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)x_C, lda, (std::complex<float>*)y_C, incx, std::complex<float>(beta_C.x(),beta_C.y()), (std::complex<float>*)result_C, incy).wait();
  cublasCgemv('N', m, n, alpha_C, x_C, lda, y_C, incx, beta_C, result_C, incy);
  // CHECK:oneapi::mkl::blas::gemv(*dpct::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, m, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)x_Z, lda, (std::complex<double>*)y_Z, incx, std::complex<double>(beta_Z.x(),beta_Z.y()), (std::complex<double>*)result_Z, incy).wait();
  cublasZgemv('N', m, n, alpha_Z, x_Z, lda, y_Z, incx, beta_Z, result_Z, incy);

  // CHECK:oneapi::mkl::blas::ger(*dpct::get_current_device().get_saved_queue(), m, n, alpha_S, x_S, incx, y_S, incy, result_S, lda).wait();
  cublasSger(m, n, alpha_S, x_S, incx, y_S, incy, result_S, lda);
  // CHECK:oneapi::mkl::blas::ger(*dpct::get_current_device().get_saved_queue(), m, n, alpha_D, x_D, incx, y_D, incy, result_D, lda).wait();
  cublasDger(m, n, alpha_D, x_D, incx, y_D, incy, result_D, lda);
  // CHECK:oneapi::mkl::blas::geru(*dpct::get_current_device().get_saved_queue(), m, n, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)result_C, lda).wait();
  cublasCgeru(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK:oneapi::mkl::blas::gerc(*dpct::get_current_device().get_saved_queue(), m, n, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)result_C, lda).wait();
  cublasCgerc(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK:oneapi::mkl::blas::geru(*dpct::get_current_device().get_saved_queue(), m, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)result_Z, lda).wait();
  cublasZgeru(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);
  // CHECK:oneapi::mkl::blas::gerc(*dpct::get_current_device().get_saved_queue(), m, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)result_Z, lda).wait();
  cublasZgerc(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);

  //level 3

  //CHECK:oneapi::mkl::blas::gemm(*dpct::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n, alpha_S, A_S, n, B_S, n, beta_S, C_S, n).wait();
  cublasSgemm('N', 'N', n, n, n, alpha_S, A_S, n, B_S, n, beta_S, C_S, n);
  //CHECK:oneapi::mkl::blas::gemm(*dpct::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n, alpha_D, A_D, n, B_D, n, beta_D, C_D, n).wait();
  cublasDgemm('N', 'N', n, n, n, alpha_D, A_D, n, B_D, n, beta_D, C_D, n);
  //CHECK:oneapi::mkl::blas::gemm(*dpct::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)A_C, n, (std::complex<float>*)B_C, n, std::complex<float>(beta_C.x(),beta_C.y()), (std::complex<float>*)C_C, n).wait();
  cublasCgemm('N', 'N', n, n, n, alpha_C, A_C, n, B_C, n, beta_C, C_C, n);
  //CHECK:oneapi::mkl::blas::gemm(*dpct::get_current_device().get_saved_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)A_Z, n, (std::complex<double>*)B_Z, n, std::complex<double>(beta_Z.x(),beta_Z.y()), (std::complex<double>*)C_Z, n).wait();
  cublasZgemm('N', 'N', n, n, n, alpha_Z, A_Z, n, B_Z, n, beta_Z, C_Z, n);

  //CHECK:oneapi::mkl::blas::trmm(*dpct::get_current_device().get_saved_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, n, alpha_S, A_S, n, C_S, n).wait();
  cublasStrmm('L', 'L', 'N', 'N', n, n, alpha_S, A_S, n, C_S, n);
  //CHECK:oneapi::mkl::blas::trmm(*dpct::get_current_device().get_saved_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, n, alpha_D, A_D, n, C_D, n).wait();
  cublasDtrmm('L', 'L', 'N', 'N', n, n, alpha_D, A_D, n, C_D, n);
  //CHECK:oneapi::mkl::blas::trmm(*dpct::get_current_device().get_saved_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, n, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)A_C, n,  (std::complex<float>*)C_C, n).wait();
  cublasCtrmm('L', 'L', 'N', 'N', n, n, alpha_C, A_C, n, C_C, n);
  //CHECK:oneapi::mkl::blas::trmm(*dpct::get_current_device().get_saved_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, n, n, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)A_Z, n,  (std::complex<double>*)C_Z, n).wait();
  cublasZtrmm('L', 'L', 'N', 'N', n, n, alpha_Z, A_Z, n, C_Z, n);
}

// Because the return value of origin API is the result value, not the status, so keep using lambda here.
//CHECK:int foo(){
//CHECK-NEXT:  return [&](){
//CHECK-NEXT:  int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
//CHECK-NEXT:  oneapi::mkl::blas::iamax(*dpct::get_current_device().get_saved_queue(), n, (std::complex<double>*)x_Z, incx, res_temp_ptr_ct{{[0-9]+}}).wait();
//CHECK-NEXT:  int64_t res_temp_val_ct{{[0-9]+}} = *res_temp_ptr_ct{{[0-9]+}};
//CHECK-NEXT:  sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
//CHECK-NEXT:  return res_temp_val_ct{{[0-9]+}};
//CHECK-NEXT:  }();
//CHECK-NEXT:}
int foo(){
  return cublasIzamax(n, x_Z, incx);
}
