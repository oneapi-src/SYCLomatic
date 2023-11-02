// RUN: dpct --format-range=none -out-root %T/cublas-usm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-usm/cublas-usm.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

cublasHandle_t handle;
int N = 275;
float *h_a, *h_b, *h_c;
const float *d_A_S;
const float *d_B_S;
float *d_C_S;
float alpha_S = 1.0f;
float beta_S = 0.0f;
int trans0 = 0;
int trans1 = 1;
int trans2 = 2;
int fill0 = 0;
int side0 = 0;
int diag0 = 0;
int *result = 0;
const float *x_S = 0;
const float *y_S = 0;

const double *d_A_D;
const double  *d_B_D;
double  *d_C_D;
double alpha_D;
double beta_D;
const double *x_D;
const double *y_D;

const float2 *d_A_C;
const float2  *d_B_C;
float2  *d_C_C;
float2 alpha_C;
float2 beta_C;
const float2 *x_C;
const float2 *y_C;

const double2 *d_A_Z;
const double2  *d_B_Z;
double2  *d_C_Z;
double2 alpha_Z;
double2 beta_Z;
const double2 *x_Z;
const double2 *y_Z;

float* result_S;
double* result_D;
float2* result_C;
double2* result_Z;

int incx, incy, lda, ldb, ldc;

int main() {

  //CHECK:/*
  //CHECK-NEXT:DPCT1018:{{[0-9]+}}: The cublasSetVector was migrated, but due to parameter 11111 equals to parameter 11111 but greater than 1, the generated code performance may be sub-optimal.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int a = DPCT_CHECK_ERROR(dpct::matrix_mem_copy((void*)d_C_S, (void*)h_a, 11111, 11111, 1, 10, sizeof(float)));
  //CHECK-NEXT:dpct::matrix_mem_copy((void*)d_C_S, (void*)h_b, 1, 1, 1, 10, sizeof(float));
  //CHECK-NEXT:dpct::matrix_mem_copy((void*)d_C_S, (void*)h_c, 1, 1, 1, 10, sizeof(float));
  //CHECK-NEXT:a = DPCT_CHECK_ERROR(dpct::matrix_mem_copy((void*)d_C_S, (void*)h_a, 100, 100, 100, 100, 10000));
  int a = cublasSetVector(10, sizeof(float), h_a, 11111, d_C_S, 11111);
  cublasSetVector(10, sizeof(float), h_b, 1, d_C_S, 1);
  cublasSetVector(10, sizeof(float), h_c, 1, d_C_S, 1);
  a = cublasSetMatrix(100, 100, 10000, h_a, 100, d_C_S, 100);


  //CHECK: int mode = 1;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasGetPointerMode was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasSetPointerMode was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_DEVICE;
  cublasGetPointerMode(handle, &mode);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

  //level 1

  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, q_ct1);
  //CHECK-NEXT:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamax(handle->get_queue(), N, x_S, N, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait());
  //CHECK-NEXT:int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:dpct::dpct_memcpy(result, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  a = cublasIsamax(handle, N, x_S, N, result);
  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, q_ct1);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::iamax(handle->get_queue(), N, x_D, N, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait();
  //CHECK-NEXT:int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:dpct::dpct_memcpy(result, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  cublasIdamax(handle, N, x_D, N, result);
  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, q_ct1);
  //CHECK-NEXT:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamax(handle->get_queue(), N, (std::complex<float>*)x_C, N, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait());
  //CHECK-NEXT:int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:dpct::dpct_memcpy(result, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  a = cublasIcamax(handle, N, x_C, N, result);
  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, q_ct1);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::iamax(handle->get_queue(), N, (std::complex<double>*)x_Z, N, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait();
  //CHECK-NEXT:int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:dpct::dpct_memcpy(result, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  cublasIzamax(handle, N, x_Z, N, result);

  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, q_ct1);
  //CHECK-NEXT:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamin(handle->get_queue(), N, x_S, N, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait());
  //CHECK-NEXT:int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:dpct::dpct_memcpy(result, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  a = cublasIsamin(handle, N, x_S, N, result);
  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, q_ct1);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::iamin(handle->get_queue(), N, x_D, N, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait();
  //CHECK-NEXT:int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:dpct::dpct_memcpy(result, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  cublasIdamin(handle, N, x_D, N, result);
  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, q_ct1);
  //CHECK-NEXT:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamin(handle->get_queue(), N, (std::complex<float>*)x_C, N, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait());
  //CHECK-NEXT:int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:dpct::dpct_memcpy(result, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  a = cublasIcamin(handle, N, x_C, N, result);
  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, q_ct1);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::iamin(handle->get_queue(), N, (std::complex<double>*)x_Z, N, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait();
  //CHECK-NEXT:int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:dpct::dpct_memcpy(result, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  cublasIzamin(handle, N, x_Z, N, result);

  //CHECK:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rotm(handle->get_queue(), N, d_C_S, N, d_C_S, N, const_cast<float*>(x_S)));
  a = cublasSrotm(handle, N, d_C_S, N, d_C_S, N, x_S);
  //CHECK:oneapi::mkl::blas::column_major::rotm(handle->get_queue(), N, d_C_D, N, d_C_D, N, const_cast<double*>(x_D));
  cublasDrotm(handle, N, d_C_D, N, d_C_D, N, x_D);

  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::copy(handle->get_queue(), N, x_S, incx, d_C_S, incy));
  a = cublasScopy(handle, N, x_S, incx, d_C_S, incy);
  // CHECK:oneapi::mkl::blas::column_major::copy(handle->get_queue(), N, x_D, incx, d_C_D, incy);
  cublasDcopy(handle, N, x_D, incx, d_C_D, incy);
  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::copy(handle->get_queue(), N, (std::complex<float>*)x_C, incx, (std::complex<float>*)d_C_C, incy));
  a = cublasCcopy(handle, N, x_C, incx, d_C_C, incy);
  // CHECK:oneapi::mkl::blas::column_major::copy(handle->get_queue(), N, (std::complex<double>*)x_Z, incx, (std::complex<double>*)d_C_Z, incy);
  cublasZcopy(handle, N, x_Z, incx, d_C_Z, incy);


  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(handle->get_queue(), N, alpha_S, x_S, incx, result_S, incy));
  a = cublasSaxpy(handle, N, &alpha_S, x_S, incx, result_S, incy);
  // CHECK:oneapi::mkl::blas::column_major::axpy(handle->get_queue(), N, alpha_D, x_D, incx, result_D, incy);
  cublasDaxpy(handle, N, &alpha_D, x_D, incx, result_D, incy);
  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(handle->get_queue(), N, std::complex<float>(alpha_C.x(), alpha_C.y()), (std::complex<float>*)x_C, incx, (std::complex<float>*)result_C, incy));
  a = cublasCaxpy(handle, N, &alpha_C, x_C, incx, result_C, incy);
  // CHECK:oneapi::mkl::blas::column_major::axpy(handle->get_queue(), N, std::complex<double>(alpha_Z.x(), alpha_Z.y()), (std::complex<double>*)x_Z, incx, (std::complex<double>*)result_Z, incy);
  cublasZaxpy(handle, N, &alpha_Z, x_Z, incx, result_Z, incy);

  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(handle->get_queue(), N, alpha_S, result_S, incx));
  a = cublasSscal(handle, N, &alpha_S, result_S, incx);
  // CHECK:oneapi::mkl::blas::column_major::scal(handle->get_queue(), N, alpha_D, result_D, incx);
  cublasDscal(handle, N, &alpha_D, result_D, incx);
  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::scal(handle->get_queue(), N, std::complex<float>(alpha_C.x(), alpha_C.y()), (std::complex<float>*)result_C, incx));
  a = cublasCscal(handle, N, &alpha_C, result_C, incx);
  // CHECK:oneapi::mkl::blas::column_major::scal(handle->get_queue(), N, std::complex<double>(alpha_Z.x(), alpha_Z.y()), (std::complex<double>*)result_Z, incx);
  cublasZscal(handle, N, &alpha_Z, result_Z, incx);

  // CHECK: float* res_temp_ptr_ct{{[0-9]+}} = result_S;
  // CHECK-NEXT: if(sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, q_ct1);
  // CHECK-NEXT: }
  // CHECK-NEXT: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), N, x_S, incx, res_temp_ptr_ct{{[0-9]+}}));
  // CHECK-NEXT: if(sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *result_S = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  a = cublasSnrm2(handle, N, x_S, incx, result_S);
  // CHECK: double* res_temp_ptr_ct{{[0-9]+}} = result_D;
  // CHECK-NEXT: if(sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, q_ct1);
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), N, x_D, incx, res_temp_ptr_ct{{[0-9]+}});
  // CHECK-NEXT: if(sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *result_D = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  cublasDnrm2(handle, N, x_D, incx, result_D);
  // CHECK: float* res_temp_ptr_ct{{[0-9]+}} = result_S;
  // CHECK-NEXT: if(sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, q_ct1);
  // CHECK-NEXT: }
  // CHECK-NEXT: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), N, (std::complex<float>*)x_C, incx, res_temp_ptr_ct{{[0-9]+}}));
  // CHECK-NEXT: if(sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *result_S = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  a = cublasScnrm2(handle, N, x_C, incx, result_S);
  // CHECK: double* res_temp_ptr_ct{{[0-9]+}} = result_D;
  // CHECK-NEXT: if(sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, q_ct1);
  // CHECK-NEXT: }
  // CHECK:oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), N, (std::complex<double>*)x_Z, incx, res_temp_ptr_ct{{[0-9]+}});
  // CHECK-NEXT: if(sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *result_D = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  cublasDznrm2(handle, N, x_Z, incx, result_D);

  // CHECK: float* res_temp_ptr_ct{{[0-9]+}} = result_S;
  // CHECK-NEXT: if(sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, q_ct1);
  // CHECK-NEXT: }
  // CHECK-NEXT: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::asum(handle->get_queue(), N, x_S, incx, res_temp_ptr_ct{{[0-9]+}}));
  // CHECK-NEXT: if(sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *result_S = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  a = cublasSasum(handle, N, x_S, incx, result_S);
  // CHECK: double* res_temp_ptr_ct{{[0-9]+}} = result_D;
  // CHECK-NEXT: if(sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, q_ct1);
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(handle->get_queue(), N, x_D, incx, res_temp_ptr_ct{{[0-9]+}});
  // CHECK-NEXT: if(sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *result_D = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  cublasDasum(handle, N, x_D, incx, result_D);
  // CHECK: float* res_temp_ptr_ct{{[0-9]+}} = result_S;
  // CHECK-NEXT: if(sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, q_ct1);
  // CHECK-NEXT: }
  // CHECK-NEXT: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::asum(handle->get_queue(), N, (std::complex<float>*)x_C, incx, res_temp_ptr_ct{{[0-9]+}}));
  // CHECK-NEXT: if(sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *result_S = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  a = cublasScasum(handle, N, x_C, incx, result_S);
  // CHECK: double* res_temp_ptr_ct{{[0-9]+}} = result_D;
  // CHECK-NEXT: if(sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, q_ct1);
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::asum(handle->get_queue(), N, (std::complex<double>*)x_Z, incx, res_temp_ptr_ct{{[0-9]+}});
  // CHECK-NEXT: if(sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *result_D = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  cublasDzasum(handle, N, x_Z, incx, result_D);

  float *a_S, *b_S, *c_S, *s_S;
  double *a_D, *b_D, *c_D, *s_D;
  float2 *a_C, *b_C, *s_C;
  double2 *a_Z, *b_Z, *s_Z;

  // CHECK: float* a_ct{{[0-9]+}} = a_S;
  // CHECK-NEXT: float* b_ct{{[0-9]+}} = b_S;
  // CHECK-NEXT: float* c_ct{{[0-9]+}} = c_S;
  // CHECK-NEXT: float* s_ct{{[0-9]+}} = s_S;
  // CHECK-NEXT: if(sycl::get_pointer_type(a_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   a_ct{{[0-9]+}} = sycl::malloc_shared<float>(4, q_ct1);
  // CHECK-NEXT:   b_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 1;
  // CHECK-NEXT:   c_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 2;
  // CHECK-NEXT:   s_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 3;
  // CHECK-NEXT:   *a_ct{{[0-9]+}} = *a_S;
  // CHECK-NEXT:   *b_ct{{[0-9]+}} = *b_S;
  // CHECK-NEXT:   *c_ct{{[0-9]+}} = *c_S;
  // CHECK-NEXT:   *s_ct{{[0-9]+}} = *s_S;
  // CHECK-NEXT: }
  // CHECK-NEXT: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rotg(handle->get_queue(), a_ct{{[0-9]+}}, b_ct{{[0-9]+}}, c_ct{{[0-9]+}}, s_ct{{[0-9]+}}));
  // CHECK-NEXT: if(sycl::get_pointer_type(a_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *a_S = *a_ct{{[0-9]+}};
  // CHECK-NEXT:   *b_S = *b_ct{{[0-9]+}};
  // CHECK-NEXT:   *c_S = *c_ct{{[0-9]+}};
  // CHECK-NEXT:   *s_S = *s_ct{{[0-9]+}};
  // CHECK-NEXT:   sycl::free(a_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  a = cublasSrotg(handle, a_S, b_S, c_S, s_S);
  // CHECK: double* a_ct{{[0-9]+}} = a_D;
  // CHECK-NEXT: double* b_ct{{[0-9]+}} = b_D;
  // CHECK-NEXT: double* c_ct{{[0-9]+}} = c_D;
  // CHECK-NEXT: double* s_ct{{[0-9]+}} = s_D;
  // CHECK-NEXT: if(sycl::get_pointer_type(a_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   a_ct{{[0-9]+}} = sycl::malloc_shared<double>(4, q_ct1);
  // CHECK-NEXT:   b_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 1;
  // CHECK-NEXT:   c_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 2;
  // CHECK-NEXT:   s_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 3;
  // CHECK-NEXT:   *a_ct{{[0-9]+}} = *a_D;
  // CHECK-NEXT:   *b_ct{{[0-9]+}} = *b_D;
  // CHECK-NEXT:   *c_ct{{[0-9]+}} = *c_D;
  // CHECK-NEXT:   *s_ct{{[0-9]+}} = *s_D;
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotg(handle->get_queue(), a_ct{{[0-9]+}}, b_ct{{[0-9]+}}, c_ct{{[0-9]+}}, s_ct{{[0-9]+}});
  // CHECK-NEXT: if(sycl::get_pointer_type(a_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *a_D = *a_ct{{[0-9]+}};
  // CHECK-NEXT:   *b_D = *b_ct{{[0-9]+}};
  // CHECK-NEXT:   *c_D = *c_ct{{[0-9]+}};
  // CHECK-NEXT:   *s_D = *s_ct{{[0-9]+}};
  // CHECK-NEXT:   sycl::free(a_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  cublasDrotg(handle, a_D, b_D, c_D, s_D);
  // CHECK: sycl::float2* a_ct{{[0-9]+}} = a_C;
  // CHECK-NEXT: sycl::float2* b_ct{{[0-9]+}} = b_C;
  // CHECK-NEXT: float* c_ct{{[0-9]+}} = c_S;
  // CHECK-NEXT: sycl::float2* s_ct{{[0-9]+}} = s_C;
  // CHECK-NEXT: if(sycl::get_pointer_type(a_C, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a_C, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   a_ct{{[0-9]+}} = sycl::malloc_shared<sycl::float2>(3, q_ct1);
  // CHECK-NEXT:   c_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, q_ct1);
  // CHECK-NEXT:   b_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 1;
  // CHECK-NEXT:   s_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 2;
  // CHECK-NEXT:   *a_ct{{[0-9]+}} = *a_C;
  // CHECK-NEXT:   *b_ct{{[0-9]+}} = *b_C;
  // CHECK-NEXT:   *c_ct{{[0-9]+}} = *c_S;
  // CHECK-NEXT:   *s_ct{{[0-9]+}} = *s_C;
  // CHECK-NEXT: }
  // CHECK-NEXT: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rotg(handle->get_queue(), (std::complex<float>*)a_ct{{[0-9]+}}, (std::complex<float>*)b_ct{{[0-9]+}}, c_ct{{[0-9]+}}, (std::complex<float>*)s_ct{{[0-9]+}}));
  // CHECK-NEXT: if(sycl::get_pointer_type(a_C, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a_C, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *a_C = *a_ct{{[0-9]+}};
  // CHECK-NEXT:   *b_C = *b_ct{{[0-9]+}};
  // CHECK-NEXT:   *c_S = *c_ct{{[0-9]+}};
  // CHECK-NEXT:   *s_C = *s_ct{{[0-9]+}};
  // CHECK-NEXT:   sycl::free(a_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:   sycl::free(c_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  a = cublasCrotg(handle, a_C, b_C, c_S, s_C);
  // CHECK: sycl::double2* a_ct{{[0-9]+}} = a_Z;
  // CHECK-NEXT: sycl::double2* b_ct{{[0-9]+}} = b_Z;
  // CHECK-NEXT: double* c_ct{{[0-9]+}} = c_D;
  // CHECK-NEXT: sycl::double2* s_ct{{[0-9]+}} = s_Z;
  // CHECK-NEXT: if(sycl::get_pointer_type(a_Z, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a_Z, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   a_ct{{[0-9]+}} = sycl::malloc_shared<sycl::double2>(3, q_ct1);
  // CHECK-NEXT:   c_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, q_ct1);
  // CHECK-NEXT:   b_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 1;
  // CHECK-NEXT:   s_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 2;
  // CHECK-NEXT:   *a_ct{{[0-9]+}} = *a_Z;
  // CHECK-NEXT:   *b_ct{{[0-9]+}} = *b_Z;
  // CHECK-NEXT:   *c_ct{{[0-9]+}} = *c_D;
  // CHECK-NEXT:   *s_ct{{[0-9]+}} = *s_Z;
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotg(handle->get_queue(), (std::complex<double>*)a_ct{{[0-9]+}}, (std::complex<double>*)b_ct{{[0-9]+}}, c_ct{{[0-9]+}}, (std::complex<double>*)s_ct{{[0-9]+}});
  // CHECK-NEXT: if(sycl::get_pointer_type(a_Z, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a_Z, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *a_Z = *a_ct{{[0-9]+}};
  // CHECK-NEXT:   *b_Z = *b_ct{{[0-9]+}};
  // CHECK-NEXT:   *c_D = *c_ct{{[0-9]+}};
  // CHECK-NEXT:   *s_Z = *s_ct{{[0-9]+}};
  // CHECK-NEXT:   sycl::free(a_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:   sycl::free(c_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  cublasZrotg(handle, a_Z, b_Z, c_D, s_Z);

  const float *y1_S;
  const double *y1_D;
  // CHECK: float* d1_ct{{[0-9]+}} = a_S;
  // CHECK-NEXT: float* d2_ct{{[0-9]+}} = b_S;
  // CHECK-NEXT: float* x1_ct{{[0-9]+}} = c_S;
  // CHECK-NEXT: float* param_ct{{[0-9]+}} = s_S;
  // CHECK-NEXT: if(sycl::get_pointer_type(a_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   d1_ct{{[0-9]+}} = sycl::malloc_shared<float>(8, q_ct1);
  // CHECK-NEXT:   d2_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 1;
  // CHECK-NEXT:   x1_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 2;
  // CHECK-NEXT:   param_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 3;
  // CHECK-NEXT:   *d1_ct{{[0-9]+}} = *a_S;
  // CHECK-NEXT:   *d2_ct{{[0-9]+}} = *b_S;
  // CHECK-NEXT:   *x1_ct{{[0-9]+}} = *c_S;
  // CHECK-NEXT: }
  // CHECK-NEXT: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::rotmg(handle->get_queue(), d1_ct{{[0-9]+}}, d2_ct{{[0-9]+}}, x1_ct{{[0-9]+}}, dpct::get_value(y1_S, handle->get_queue()), param_ct{{[0-9]+}}));
  // CHECK-NEXT: if(sycl::get_pointer_type(a_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *a_S = *d1_ct{{[0-9]+}};
  // CHECK-NEXT:   *b_S = *d2_ct{{[0-9]+}};
  // CHECK-NEXT:   *c_S = *x1_ct{{[0-9]+}};
  // CHECK-NEXT:   q_ct1.memcpy(s_S, param_ct{{[0-9]+}}, sizeof(float)*5).wait();
  // CHECK-NEXT:   sycl::free(d1_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  a = cublasSrotmg(handle, a_S, b_S, c_S, y1_S, s_S);
  // CHECK: double* d1_ct{{[0-9]+}} = a_D;
  // CHECK-NEXT: double* d2_ct{{[0-9]+}} = b_D;
  // CHECK-NEXT: double* x1_ct{{[0-9]+}} = c_D;
  // CHECK-NEXT: double* param_ct{{[0-9]+}} = s_D;
  // CHECK-NEXT: if(sycl::get_pointer_type(a_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   d1_ct{{[0-9]+}} = sycl::malloc_shared<double>(8, q_ct1);
  // CHECK-NEXT:   d2_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 1;
  // CHECK-NEXT:   x1_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 2;
  // CHECK-NEXT:   param_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 3;
  // CHECK-NEXT:   *d1_ct{{[0-9]+}} = *a_D;
  // CHECK-NEXT:   *d2_ct{{[0-9]+}} = *b_D;
  // CHECK-NEXT:   *x1_ct{{[0-9]+}} = *c_D;
  // CHECK-NEXT: }
  // CHECK-NEXT: oneapi::mkl::blas::column_major::rotmg(handle->get_queue(), d1_ct{{[0-9]+}}, d2_ct{{[0-9]+}}, x1_ct{{[0-9]+}}, dpct::get_value(y1_D, handle->get_queue()), param_ct{{[0-9]+}});
  // CHECK-NEXT: if(sycl::get_pointer_type(a_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:   handle->get_queue().wait();
  // CHECK-NEXT:   *a_D = *d1_ct{{[0-9]+}};
  // CHECK-NEXT:   *b_D = *d2_ct{{[0-9]+}};
  // CHECK-NEXT:   *c_D = *x1_ct{{[0-9]+}};
  // CHECK-NEXT:   q_ct1.memcpy(s_D, param_ct{{[0-9]+}}, sizeof(double)*5).wait();
  // CHECK-NEXT:   sycl::free(d1_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT: }
  cublasDrotmg(handle, a_D, b_D, c_D, y1_D, s_D);

  // CHECK:float* res_temp_ptr_ct{{[0-9]+}} = result_S;
  // CHECK-NEXT:if(sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, q_ct1);
  // CHECK-NEXT:}
  // CHECK-NEXT:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dot(handle->get_queue(), N, x_S, incx, y_S, incy, res_temp_ptr_ct{{[0-9]+}}));
  // CHECK-NEXT:if(sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  handle->get_queue().wait();
  // CHECK-NEXT:  *result_S = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:  sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:}
  a = cublasSdot(handle, N, x_S, incx, y_S, incy, result_S);
  // CHECK:double* res_temp_ptr_ct{{[0-9]+}} = result_D;
  // CHECK-NEXT:if(sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, q_ct1);
  // CHECK-NEXT:}
  // CHECK-NEXT:oneapi::mkl::blas::column_major::dot(handle->get_queue(), N, x_D, incx, y_D, incy, res_temp_ptr_ct{{[0-9]+}});
  // CHECK-NEXT:if(sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  handle->get_queue().wait();
  // CHECK-NEXT:  *result_D = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:  sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:}
  cublasDdot(handle, N, x_D, incx, y_D, incy, result_D);

  // CHECK:sycl::float2* res_temp_ptr_ct{{[0-9]+}} = result_C;
  // CHECK-NEXT:if(sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::float2>(1, q_ct1);
  // CHECK-NEXT:}
  // CHECK-NEXT:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dotc(handle->get_queue(), N, (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)res_temp_ptr_ct{{[0-9]+}}));
  // CHECK-NEXT:if(sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  handle->get_queue().wait();
  // CHECK-NEXT:  *result_C = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:  sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:}
  a = cublasCdotc(handle, N, x_C, incx, y_C, incy, result_C);
  // CHECK:sycl::double2* res_temp_ptr_ct{{[0-9]+}} = result_Z;
  // CHECK-NEXT:if(sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::double2>(1, q_ct1);
  // CHECK-NEXT:}
  // CHECK-NEXT:oneapi::mkl::blas::column_major::dotc(handle->get_queue(), N, (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)res_temp_ptr_ct{{[0-9]+}});
  // CHECK-NEXT:if(sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  handle->get_queue().wait();
  // CHECK-NEXT:  *result_Z = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:  sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:}
  cublasZdotc(handle, N, x_Z, incx, y_Z, incy, result_Z);

  // CHECK:sycl::float2* res_temp_ptr_ct{{[0-9]+}} = result_C;
  // CHECK-NEXT:if(sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::float2>(1, q_ct1);
  // CHECK-NEXT:}
  // CHECK-NEXT:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dotu(handle->get_queue(), N, (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)res_temp_ptr_ct{{[0-9]+}}));
  // CHECK-NEXT:if(sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  handle->get_queue().wait();
  // CHECK-NEXT:  *result_C = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:  sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:}
  a = cublasCdotu(handle, N, x_C, incx, y_C, incy, result_C);
  // CHECK:sycl::double2* res_temp_ptr_ct{{[0-9]+}} = result_Z;
  // CHECK-NEXT:if(sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::double2>(1, q_ct1);
  // CHECK-NEXT:}
  // CHECK-NEXT:oneapi::mkl::blas::column_major::dotu(handle->get_queue(), N, (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)res_temp_ptr_ct{{[0-9]+}});
  // CHECK-NEXT:if(sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  handle->get_queue().wait();
  // CHECK-NEXT:  *result_Z = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:  sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:}
  cublasZdotu(handle, N, x_Z, incx, y_Z, incy, result_Z);

  // CHECK:float* res_temp_ptr_ct{{[0-9]+}} = result_S;
  // CHECK-NEXT:if(sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, q_ct1);
  // CHECK-NEXT:}
  // CHECK-NEXT:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dot(handle->get_queue(), N, x_S, incx, y_S, incy, res_temp_ptr_ct{{[0-9]+}}));
  // CHECK-NEXT:if(sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  handle->get_queue().wait();
  // CHECK-NEXT:  *result_S = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:  sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:}
  a = cublasSdot(handle, N, x_S, incx, y_S, incy, result_S);
  // CHECK:double* res_temp_ptr_ct{{[0-9]+}} = result_D;
  // CHECK-NEXT:if(sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, q_ct1);
  // CHECK-NEXT:}
  // CHECK-NEXT:oneapi::mkl::blas::column_major::dot(handle->get_queue(), N, x_D, incx, y_D, incy, res_temp_ptr_ct{{[0-9]+}});
  // CHECK-NEXT:if(sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  handle->get_queue().wait();
  // CHECK-NEXT:  *result_D = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:  sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:}
  cublasDdot(handle, N, x_D, incx, y_D, incy, result_D);

  // CHECK:sycl::float2* res_temp_ptr_ct{{[0-9]+}} = result_C;
  // CHECK-NEXT:if(sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::float2>(1, q_ct1);
  // CHECK-NEXT:}
  // CHECK-NEXT:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dotc(handle->get_queue(), N, (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)res_temp_ptr_ct{{[0-9]+}}));
  // CHECK-NEXT:if(sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  handle->get_queue().wait();
  // CHECK-NEXT:  *result_C = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:  sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:}
  a = cublasCdotc(handle, N, x_C, incx, y_C, incy, result_C);
  // CHECK:sycl::double2* res_temp_ptr_ct{{[0-9]+}} = result_Z;
  // CHECK-NEXT:if(sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::double2>(1, q_ct1);
  // CHECK-NEXT:}
  // CHECK-NEXT:oneapi::mkl::blas::column_major::dotc(handle->get_queue(), N, (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)res_temp_ptr_ct{{[0-9]+}});
  // CHECK-NEXT:if(sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  handle->get_queue().wait();
  // CHECK-NEXT:  *result_Z = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:  sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:}
  cublasZdotc(handle, N, x_Z, incx, y_Z, incy, result_Z);

  // CHECK:sycl::float2* res_temp_ptr_ct{{[0-9]+}} = result_C;
  // CHECK-NEXT:if(sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::float2>(1, q_ct1);
  // CHECK-NEXT:}
  // CHECK-NEXT:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dotu(handle->get_queue(), N, (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)res_temp_ptr_ct{{[0-9]+}}));
  // CHECK-NEXT:if(sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_C, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  handle->get_queue().wait();
  // CHECK-NEXT:  *result_C = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:  sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:}
  a = cublasCdotu(handle, N, x_C, incx, y_C, incy, result_C);
  // CHECK:sycl::double2* res_temp_ptr_ct{{[0-9]+}} = result_Z;
  // CHECK-NEXT:if(sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::double2>(1, q_ct1);
  // CHECK-NEXT:}
  // CHECK-NEXT:oneapi::mkl::blas::column_major::dotu(handle->get_queue(), N, (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)res_temp_ptr_ct{{[0-9]+}});
  // CHECK-NEXT:if(sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_Z, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
  // CHECK-NEXT:  handle->get_queue().wait();
  // CHECK-NEXT:  *result_Z = *res_temp_ptr_ct{{[0-9]+}};
  // CHECK-NEXT:  sycl::free(res_temp_ptr_ct{{[0-9]+}}, q_ct1);
  // CHECK-NEXT:}
  cublasZdotu(handle, N, x_Z, incx, y_Z, incy, result_Z);

  //level 2

  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemv(handle->get_queue(), trans2==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans2, N, N, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy));
  a = cublasSgemv(handle, (cublasOperation_t)trans2, N, N, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  // CHECK:oneapi::mkl::blas::column_major::gemv(handle->get_queue(), oneapi::mkl::transpose::nontrans, N, N, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);
  cublasDgemv(handle, CUBLAS_OP_N, N, N, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemv(handle->get_queue(), trans2==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans2, N, N, std::complex<float>(alpha_C.x(), alpha_C.y()), (std::complex<float>*)x_C, lda, (std::complex<float>*)y_C, incx, std::complex<float>(beta_C.x(), beta_C.y()), (std::complex<float>*)result_C, incy));
  a = cublasCgemv(handle, (cublasOperation_t)trans2, N, N, &alpha_C, x_C, lda, y_C, incx, &beta_C, result_C, incy);
  // CHECK:oneapi::mkl::blas::column_major::gemv(handle->get_queue(), oneapi::mkl::transpose::nontrans, N, N, std::complex<double>(alpha_Z.x(), alpha_Z.y()), (std::complex<double>*)x_Z, lda, (std::complex<double>*)y_Z, incx, std::complex<double>(beta_Z.x(), beta_Z.y()), (std::complex<double>*)result_Z, incy);
  cublasZgemv(handle, CUBLAS_OP_N, N, N, &alpha_Z, x_Z, lda, y_Z, incx, &beta_Z, result_Z, incy);

  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::ger(handle->get_queue(), N, N, alpha_S, x_S, incx, y_S, incy, result_S, lda));
  a = cublasSger(handle, N, N, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  // CHECK:oneapi::mkl::blas::column_major::ger(handle->get_queue(), N, N, alpha_D, x_D, incx, y_D, incy, result_D, lda);
  cublasDger(handle, N, N, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::geru(handle->get_queue(), N, N, std::complex<float>(alpha_C.x(), alpha_C.y()), (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)result_C, lda));
  a = cublasCgeru(handle, N, N, &alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK:oneapi::mkl::blas::column_major::gerc(handle->get_queue(), N, N, std::complex<float>(alpha_C.x(), alpha_C.y()), (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)result_C, lda);
  cublasCgerc(handle, N, N, &alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::geru(handle->get_queue(), N, N, std::complex<double>(alpha_Z.x(), alpha_Z.y()), (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)result_Z, lda));
  a = cublasZgeru(handle, N, N, &alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);
  // CHECK:oneapi::mkl::blas::column_major::gerc(handle->get_queue(), N, N, std::complex<double>(alpha_Z.x(), alpha_Z.y()), (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)result_Z, lda);
  cublasZgerc(handle, N, N, &alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);

  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(handle->get_queue(), fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, N, std::complex<float>(alpha_C.x(), alpha_C.y()), (std::complex<float>*)x_C, lda, (std::complex<float>*)y_C, incx, std::complex<float>(beta_C.x(), beta_C.y()), (std::complex<float>*)result_C, incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symv(handle->get_queue(), oneapi::mkl::uplo::upper, N, std::complex<float>(alpha_C.x(), alpha_C.y()), (std::complex<float>*)x_C, lda, (std::complex<float>*)y_C, incx, std::complex<float>(beta_C.x(), beta_C.y()), (std::complex<float>*)result_C, incy);
  a = cublasCsymv(handle, (cublasFillMode_t)fill0, N, &alpha_C, x_C, lda, y_C, incx, &beta_C, result_C, incy);
  cublasCsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_C, x_C, lda, y_C, incx, &beta_C, result_C, incy);

  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::symv(handle->get_queue(), fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, N, std::complex<double>(alpha_Z.x(), alpha_Z.y()), (std::complex<double>*)x_Z, lda, (std::complex<double>*)y_Z, incx, std::complex<double>(beta_Z.x(), beta_Z.y()), (std::complex<double>*)result_Z, incy));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::symv(handle->get_queue(), oneapi::mkl::uplo::upper, N, std::complex<double>(alpha_Z.x(), alpha_Z.y()), (std::complex<double>*)x_Z, lda, (std::complex<double>*)y_Z, incx, std::complex<double>(beta_Z.x(), beta_Z.y()), (std::complex<double>*)result_Z, incy);
  a = cublasZsymv(handle, (cublasFillMode_t)fill0, N, &alpha_Z, x_Z, lda, y_Z, incx, &beta_Z, result_Z, incy);
  cublasZsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_Z, x_Z, lda, y_Z, incx, &beta_Z, result_Z, incy);

  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr(handle->get_queue(), fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, N, std::complex<float>(alpha_C.x(), alpha_C.y()), (std::complex<float>*)x_C, incx, (std::complex<float>*)result_C, lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr(handle->get_queue(), oneapi::mkl::uplo::upper, N, std::complex<float>(alpha_C.x(), alpha_C.y()), (std::complex<float>*)x_C, incx, (std::complex<float>*)result_C, lda);
  a = cublasCsyr(handle, (cublasFillMode_t)fill0, N, &alpha_C, x_C, incx, result_C, lda);
  cublasCsyr(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_C, x_C, incx, result_C, lda);

  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr(handle->get_queue(), fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, N, std::complex<double>(alpha_Z.x(), alpha_Z.y()), (std::complex<double>*)x_Z, incx, (std::complex<double>*)result_Z, lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr(handle->get_queue(), oneapi::mkl::uplo::upper, N, std::complex<double>(alpha_Z.x(), alpha_Z.y()), (std::complex<double>*)x_Z, incx, (std::complex<double>*)result_Z, lda);
  a = cublasZsyr(handle, (cublasFillMode_t)fill0, N, &alpha_Z, x_Z, incx, result_Z, lda);
  cublasZsyr(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_Z, x_Z, incx, result_Z, lda);

  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2(handle->get_queue(), fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, N, std::complex<float>(alpha_C.x(), alpha_C.y()), (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)result_C, lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2(handle->get_queue(), oneapi::mkl::uplo::upper, N, std::complex<float>(alpha_C.x(), alpha_C.y()), (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)result_C, lda);
  a = cublasCsyr2(handle, (cublasFillMode_t)fill0, N, &alpha_C, x_C, incx, y_C, incy, result_C, lda);
  cublasCsyr2(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_C, x_C, incx, y_C, incy, result_C, lda);

  // CHECK: a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::syr2(handle->get_queue(), fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, N, std::complex<double>(alpha_Z.x(), alpha_Z.y()), (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)result_Z, lda));
  // CHECK-NEXT: oneapi::mkl::blas::column_major::syr2(handle->get_queue(), oneapi::mkl::uplo::upper, N, std::complex<double>(alpha_Z.x(), alpha_Z.y()), (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)result_Z, lda);
  a = cublasZsyr2(handle, (cublasFillMode_t)fill0, N, &alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);
  cublasZsyr2(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);

  //level 3

  __half *d_A_H = 0;
  __half *d_B_H = 0;
  __half *d_C_H = 0;
  __half alpha_H;
  __half beta_H;

  //CHECK:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), dpct::get_transpose(trans0), dpct::get_transpose(trans1), N, N, N, dpct::get_value(&alpha_S, handle->get_queue()), d_A_S, N, d_B_S, N, dpct::get_value(&beta_S, handle->get_queue()), d_C_S, N));
  a = cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  //CHECK:oneapi::mkl::blas::column_major::gemm(handle->get_queue(), dpct::get_transpose(trans0), dpct::get_transpose(trans1), N, N, N, dpct::get_value(&alpha_D, handle->get_queue()), d_A_D, N, d_B_D, N, dpct::get_value(&beta_D, handle->get_queue()), d_C_D, N);
  cublasDgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  //CHECK:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, dpct::get_value(&alpha_C, handle->get_queue()), (std::complex<float>*)d_A_C, N, (std::complex<float>*)d_B_C, N, dpct::get_value(&beta_C, handle->get_queue()), (std::complex<float>*)d_C_C, N));
  a = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N);
  //CHECK:oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, dpct::get_value(&alpha_Z, handle->get_queue()), (std::complex<double>*)d_A_Z, N, (std::complex<double>*)d_B_Z, N, dpct::get_value(&beta_Z, handle->get_queue()), (std::complex<double>*)d_C_Z, N);
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, &beta_Z, d_C_Z, N);

  //CHECK:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(handle->get_queue(), trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, trans1==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans1, N, N, N, alpha_S, d_A_S, N, 16, d_B_S, N, 16, beta_S, d_C_S, N, 16, 10));
  a = cublasSgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, 16, d_B_S, N, 16, &beta_S, d_C_S, N, 16, 10);
  //CHECK:oneapi::mkl::blas::column_major::gemm_batch(handle->get_queue(), trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, trans1==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans1, N, N, N, alpha_D, d_A_D, N, 16, d_B_D, N, 16, beta_D, d_C_D, N, 16, 10);
  cublasDgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_D, d_A_D, N, 16, d_B_D, N, 16, &beta_D, d_C_D, N, 16, 10);
  //CHECK:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, std::complex<float>(alpha_C.x(), alpha_C.y()), (std::complex<float>*)d_A_C, N, 16, (std::complex<float>*)d_B_C, N, 16, std::complex<float>(beta_C.x(), beta_C.y()), (std::complex<float>*)d_C_C, N, 16, 10));
  a = cublasCgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C, d_A_C, N, 16, d_B_C, N, 16, &beta_C, d_C_C, N, 16, 10);
  //CHECK:oneapi::mkl::blas::column_major::gemm_batch(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, std::complex<double>(alpha_Z.x(), alpha_Z.y()), (std::complex<double>*)d_A_Z, N, 16, (std::complex<double>*)d_B_Z, N, 16, std::complex<double>(beta_Z.x(), beta_Z.y()), (std::complex<double>*)d_C_Z, N, 16, 10);
  cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_Z, d_A_Z, N, 16, d_B_Z, N, 16, &beta_Z, d_C_Z, N, 16, 10);
  //CHECK:oneapi::mkl::blas::column_major::gemm_batch(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, alpha_H, d_A_H, N, 16, d_B_H, N, 16, beta_H, d_C_H, N, 16, 10);
  cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_H, d_A_H, N, 16, d_B_H, N, 16, &beta_H, d_C_H, N, 16, 10);

  cublasOperation_t trans3 = CUBLAS_OP_N;
  //CHECK:a = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), trans3, trans3, N, N, N, alpha_H, d_A_H, N, d_B_H, N, beta_H, d_C_H, N));
  a = cublasHgemm(handle, trans3, trans3, N, N, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);

  // CHECK: void *alpha, *beta, *A, *B, *C;
  // CHECK-NEXT: int algo = 0;
  void *alpha, *beta, *A, *B, *C;
  cublasGemmAlgo_t algo = CUBLAS_GEMM_ALGO0;
  // CHECK: dpct::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, alpha, A, dpct::library_data_t::real_float, N, B, dpct::library_data_t::real_float, N, beta, C, dpct::library_data_t::real_float, N, dpct::library_data_t::real_float);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_32F, N, B, CUDA_R_32F, N, beta, C, CUDA_R_32F, N, CUDA_R_32F, algo);

  float2 alpha_C, beta_C;
  //CHECK: dpct::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, &alpha_S, A, dpct::library_data_t::real_half, N, B, dpct::library_data_t::real_half, N, &beta_S, C, dpct::library_data_t::real_half, N, dpct::library_data_t::real_float);
  //CHECK-NEXT: dpct::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, &alpha_S, A, dpct::library_data_t::real_half, N, B, dpct::library_data_t::real_half, N, &beta_S, C, dpct::library_data_t::real_float, N, dpct::library_data_t::real_float);
  //CHECK-NEXT: dpct::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, &alpha_S, A, dpct::library_data_t::real_float, N, B, dpct::library_data_t::real_float, N, &beta_S, C, dpct::library_data_t::real_float, N, dpct::library_data_t::real_float);
  //CHECK-NEXT: dpct::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, &alpha_C, A, dpct::library_data_t::complex_float, N, B, dpct::library_data_t::complex_float, N, &beta_C, C, dpct::library_data_t::complex_float, N, dpct::library_data_t::complex_float);
  cublasSgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_S, A, CUDA_R_16F, N, B, CUDA_R_16F, N, &beta_S, C, CUDA_R_16F, N);
  cublasSgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_S, A, CUDA_R_16F, N, B, CUDA_R_16F, N, &beta_S, C, CUDA_R_32F, N);
  cublasSgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_S, A, CUDA_R_32F, N, B, CUDA_R_32F, N, &beta_S, C, CUDA_R_32F, N);
  cublasCgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_C, A, CUDA_C_32F, N, B, CUDA_C_32F, N, &beta_C, C, CUDA_C_32F, N);

  const float** d_A_S_array;
  const float** d_B_S_array;
  float** d_C_S_array;
  const double** d_A_D_array;
  const double** d_B_D_array;
  double** d_C_D_array;
  const cuComplex** d_A_C_array = 0;
  const cuComplex** d_B_C_array = 0;
  cuComplex** d_C_C_array = 0;
  const cuDoubleComplex** d_A_Z_array = 0;
  const cuDoubleComplex** d_B_Z_array = 0;
  cuDoubleComplex** d_C_Z_array = 0;
  const __half** d_A_H_array = 0;
  const __half** d_B_H_array = 0;
  __half** d_C_H_array = 0;

  // CHECK: a = DPCT_CHECK_ERROR(dpct::gemm_batch(handle->get_queue(), trans3, trans3, N, N, N, &alpha_S, (const void**)d_A_S_array, dpct::library_data_t::real_float, N, (const void**)d_B_S_array, dpct::library_data_t::real_float, N, &beta_S, (void**)d_C_S_array, dpct::library_data_t::real_float, N, 10, dpct::library_data_t::real_float));
  // CHECK-NEXT: dpct::gemm_batch(handle->get_queue(), trans3, trans3, N, N, N, &alpha_D, (const void**)d_A_D_array, dpct::library_data_t::real_double, N, (const void**)d_B_D_array, dpct::library_data_t::real_double, N, &beta_D, (void**)d_C_D_array, dpct::library_data_t::real_double, N, 10, dpct::library_data_t::real_double);
  // CHECK-NEXT: dpct::gemm_batch(handle->get_queue(), trans3, trans3, N, N, N, &alpha_C, (const void**)d_A_C_array, dpct::library_data_t::complex_float, N, (const void**)d_B_C_array, dpct::library_data_t::complex_float, N, &beta_C, (void**)d_C_C_array, dpct::library_data_t::complex_float, N, 10, dpct::library_data_t::complex_float);
  // CHECK-NEXT: dpct::gemm_batch(handle->get_queue(), trans3, trans3, N, N, N, &alpha_Z, (const void**)d_A_Z_array, dpct::library_data_t::complex_double, N, (const void**)d_B_Z_array, dpct::library_data_t::complex_double, N, &beta_Z, (void**)d_C_Z_array, dpct::library_data_t::complex_double, N, 10, dpct::library_data_t::complex_double);
  a = cublasSgemmBatched(handle, trans3, trans3, N, N, N, &alpha_S, d_A_S_array, N, d_B_S_array, N, &beta_S, d_C_S_array, N, 10);
  cublasDgemmBatched(handle, trans3, trans3, N, N, N, &alpha_D, d_A_D_array, N, d_B_D_array, N, &beta_D, d_C_D_array, N, 10);
  cublasCgemmBatched(handle, trans3, trans3, N, N, N, &alpha_C, d_A_C_array, N, d_B_C_array, N, &beta_C, d_C_C_array, N, 10);
  cublasZgemmBatched(handle, trans3, trans3, N, N, N, &alpha_Z, d_A_Z_array, N, d_B_Z_array, N, &beta_Z, d_C_Z_array, N, 10);

  // CHECK: a = DPCT_CHECK_ERROR(dpct::trsm_batch(handle->get_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, trans3, oneapi::mkl::diag::unit, N, N, &alpha_S, (const void**)d_A_S_array, dpct::library_data_t::real_float, N, (void**)d_C_S_array, dpct::library_data_t::real_float, N, 10, dpct::library_data_t::real_float));
  // CHECK-NEXT: dpct::trsm_batch(handle->get_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, trans3, oneapi::mkl::diag::unit, N, N, &alpha_D, (const void**)d_A_D_array, dpct::library_data_t::real_double, N, (void**)d_C_D_array, dpct::library_data_t::real_double, N, 10, dpct::library_data_t::real_double);
  // CHECK-NEXT: dpct::trsm_batch(handle->get_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, trans3, oneapi::mkl::diag::unit, N, N, &alpha_C, (const void**)d_A_C_array, dpct::library_data_t::complex_float, N, (void**)d_C_C_array, dpct::library_data_t::complex_float, N, 10, dpct::library_data_t::complex_float);
  // CHECK-NEXT: dpct::trsm_batch(handle->get_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, trans3, oneapi::mkl::diag::unit, N, N, &alpha_Z, (const void**)d_A_Z_array, dpct::library_data_t::complex_double, N, (void**)d_C_Z_array, dpct::library_data_t::complex_double, N, 10, dpct::library_data_t::complex_double);
  a = cublasStrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans3, CUBLAS_DIAG_UNIT, N, N, &alpha_S, d_A_S_array, N, d_C_S_array, N, 10);
  cublasDtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans3, CUBLAS_DIAG_UNIT, N, N, &alpha_D, d_A_D_array, N, d_C_D_array, N, 10);
  cublasCtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans3, CUBLAS_DIAG_UNIT, N, N, &alpha_C, d_A_C_array, N, d_C_C_array, N, 10);
  cublasZtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans3, CUBLAS_DIAG_UNIT, N, N, &alpha_Z, d_A_Z_array, N, d_C_Z_array, N, 10);

  //CHECK:a = DPCT_CHECK_ERROR(dpct::trmm(handle->get_queue(), (oneapi::mkl::side)side0, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N));
  //CHECK-NEXT:dpct::trmm(handle->get_queue(), (oneapi::mkl::side)side0, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, N, N, &alpha_D, d_A_D, N, d_B_D, N, d_C_D, N);
  //CHECK:a = DPCT_CHECK_ERROR(dpct::trmm(handle->get_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, N, N, &alpha_C, d_A_C, N, d_B_C, N, d_C_C, N));
  //CHECK-NEXT:dpct::trmm(handle->get_queue(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, d_C_Z, N);
  a = cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N);
  cublasDtrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_D, d_A_D, N, d_B_D, N, d_C_D, N);
  a = cublasCtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, N, &alpha_C, d_A_C, N, d_B_C, N, d_C_C, N);
  cublasZtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, d_C_Z, N);

  //CHECK:a = DPCT_CHECK_ERROR(dpct::syrk(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans1), N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N));
  a = cublasSsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans1, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  //CHECK:dpct::syrk(handle->get_queue(), fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans1), N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  cublasDsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans1, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);

  // CHECK: if(int stat = DPCT_CHECK_ERROR(dpct::trmm(handle->get_queue(), (oneapi::mkl::side)side0, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N))){}
  if(int stat = cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N)){}

  // CHECK: if(int stat = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), dpct::get_transpose(trans0), dpct::get_transpose(trans1), N, N, N, dpct::get_value(&alpha_S, handle->get_queue()), d_A_S, N, d_B_S, N, dpct::get_value(&beta_S, handle->get_queue()), d_C_S, N))){}
  if(int stat = cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){}


}

// CHECK:int foo1() try {
// CHECK:  return DPCT_CHECK_ERROR(dpct::trmm(handle->get_queue(), (oneapi::mkl::side)side0, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N));
// CHECK-NEXT:}
int foo1(){
  return cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N);
}

// CHECK:int foo2() try {
// CHECK:  return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), dpct::get_transpose(trans0), dpct::get_transpose(trans1), N, N, N, dpct::get_value(&alpha_S, handle->get_queue()), d_A_S, N, d_B_S, N, dpct::get_value(&beta_S, handle->get_queue()), d_C_S, N));
// CHECK-NEXT:}
int foo2(){
  return cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
}

void foo3() {
  cublasHandle_t handle;
  float   *a_f, *b_f, *x_f, *c_f, *alpha_f, *beta_f;
  double  *a_d, *b_d, *x_d, *c_d, *alpha_d, *beta_d;
  float2  *a_c, *b_c, *x_c, *c_c, *alpha_c, *beta_c;
  double2 *a_z, *b_z, *x_z, *c_z, *alpha_z, *beta_z;

  //CHECK:dpct::syrk(handle->get_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, 2, 3, alpha_f, a_f, 3, b_f, 3, beta_f, c_f, 2);
  //CHECK-NEXT:dpct::syrk(handle->get_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, 2, 3, alpha_d, a_d, 3, b_d, 3, beta_d, c_d, 2);
  //CHECK-NEXT:dpct::syrk(handle->get_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, 2, 3, alpha_c, a_c, 3, b_c, 3, beta_c, c_c, 2);
  //CHECK-NEXT:dpct::syrk(handle->get_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, 2, 3, alpha_z, a_z, 3, b_z, 3, beta_z, c_z, 2);
  //CHECK-NEXT:dpct::herk(handle->get_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, 2, 3, alpha_c, a_c, 3, b_c, 3, beta_f, c_c, 2);
  //CHECK-NEXT:dpct::herk(handle->get_queue(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, 2, 3, alpha_z, a_z, 3, b_z, 3, beta_d, c_z, 2);
  cublasSsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, 2, 3, alpha_f, a_f, 3, b_f, 3, beta_f, c_f, 2);
  cublasDsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, 2, 3, alpha_d, a_d, 3, b_d, 3, beta_d, c_d, 2);
  cublasCsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, 2, 3, alpha_c, a_c, 3, b_c, 3, beta_c, c_c, 2);
  cublasZsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, 2, 3, alpha_z, a_z, 3, b_z, 3, beta_z, c_z, 2);
  cublasCherkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2, 3, alpha_c, a_c, 3, b_c, 3, beta_f, c_c, 2);
  cublasZherkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2, 3, alpha_z, a_z, 3, b_z, 3, beta_d, c_z, 2);

  int m, n, lda, incx, ldc;
  //CHECK:oneapi::mkl::blas::column_major::dgmm_batch(handle->get_queue(), oneapi::mkl::side::left, m, n, a_f, lda, 0, x_f, incx, 0, c_f, ldc, ldc * n, 1);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::dgmm_batch(handle->get_queue(), oneapi::mkl::side::left, m, n, a_d, lda, 0, x_d, incx, 0, c_d, ldc, ldc * n, 1);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::dgmm_batch(handle->get_queue(), oneapi::mkl::side::left, m, n, (std::complex<float>*)a_f, lda, 0, (std::complex<float>*)x_c, incx, 0, (std::complex<float>*)c_c, ldc, ldc * n, 1);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::dgmm_batch(handle->get_queue(), oneapi::mkl::side::left, m, n, (std::complex<double>*)a_z, lda, 0, (std::complex<double>*)x_z, incx, 0, (std::complex<double>*)c_z, ldc, ldc * n, 1);
  cublasSdgmm(handle, CUBLAS_SIDE_LEFT, m, n, a_f, lda, x_f, incx, c_f, ldc);
  cublasDdgmm(handle, CUBLAS_SIDE_LEFT, m, n, a_d, lda, x_d, incx, c_d, ldc);
  cublasCdgmm(handle, CUBLAS_SIDE_LEFT, m, n, (float2*)a_f, lda, x_c, incx, c_c, ldc);
  cublasZdgmm(handle, CUBLAS_SIDE_LEFT, m, n, a_z, lda, x_z, incx, c_z, ldc);

  //CHECK:oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::trans, m, n, dpct::get_value(alpha_f, handle->get_queue()), a_f, lda, dpct::get_value(beta_f, handle->get_queue()), b_f, ldb, c_f, ldc);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::trans, m, n, dpct::get_value(alpha_d, handle->get_queue()), a_d, lda, dpct::get_value(beta_d, handle->get_queue()), b_d, ldb, c_d, ldc);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::trans, m, n, dpct::get_value(alpha_c, handle->get_queue()), (std::complex<float>*)a_c, lda, dpct::get_value(beta_c, handle->get_queue()), (std::complex<float>*)b_c, ldb, (std::complex<float>*)c_c, ldc);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::trans, m, n, dpct::get_value(alpha_z, handle->get_queue()), (std::complex<double>*)a_z, lda, dpct::get_value(beta_z, handle->get_queue()), (std::complex<double>*)b_z, ldb, (std::complex<double>*)c_z, ldc);
  cublasSgeam(handle, CUBLAS_OP_C, CUBLAS_OP_T, m, n, alpha_f, a_f, lda, beta_f, b_f, ldb, c_f, ldc);
  cublasDgeam(handle, CUBLAS_OP_C, CUBLAS_OP_T, m, n, alpha_d, a_d, lda, beta_d, b_d, ldb, c_d, ldc);
  cublasCgeam(handle, CUBLAS_OP_C, CUBLAS_OP_T, m, n, alpha_c, a_c, lda, beta_c, b_c, ldb, c_c, ldc);
  cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_T, m, n, alpha_z, a_z, lda, beta_z, b_z, ldb, c_z, ldc);
}

