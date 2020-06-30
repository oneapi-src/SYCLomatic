// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-usm.dp.cpp --match-full-lines %s
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
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int a = (dpct::matrix_mem_copy((void*)d_C_S, (void*)h_a, 11111, 11111, 1, 10, sizeof(float)), 0);
  //CHECK-NEXT:dpct::matrix_mem_copy((void*)d_C_S, (void*)h_b, 1, 1, 1, 10, sizeof(float));
  //CHECK-NEXT:dpct::matrix_mem_copy((void*)d_C_S, (void*)h_c, 1, 1, 1, 10, sizeof(float));
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (dpct::matrix_mem_copy((void*)d_C_S, (void*)h_a, 100, 100, 100, 100, 10000), 0);
  int a = cublasSetVector(10, sizeof(float), h_a, 11111, d_C_S, 11111);
  cublasSetVector(10, sizeof(float), h_b, 1, d_C_S, 1);
  cublasSetVector(10, sizeof(float), h_c, 1, d_C_S, 1);
  a = cublasSetMatrix(100, 100, 10000, h_a, 100, d_C_S, 100);


  //CHECK: int mode = 1;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasGetPointerMode was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasSetPointerMode was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_DEVICE;
  cublasGetPointerMode(handle, &mode);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

  //level 1

  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::iamax(*handle, N, x_S, N, res_temp_ptr_ct{{[0-9]+}}), 0);
  //CHECK-NEXT:if(sycl::get_pointer_type(result, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result, handle->get_context())!=sycl::usm::alloc::shared) handle->wait();
  //CHECK-NEXT:*result = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  a = cublasIsamax(handle, N, x_S, N, result);
  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  //CHECK-NEXT:mkl::blas::iamax(*handle, N, x_D, N, res_temp_ptr_ct{{[0-9]+}});
  //CHECK-NEXT:if(sycl::get_pointer_type(result, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result, handle->get_context())!=sycl::usm::alloc::shared) handle->wait();
  //CHECK-NEXT:*result = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  cublasIdamax(handle, N, x_D, N, result);
  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::iamax(*handle, N, (std::complex<float>*)x_C, N, res_temp_ptr_ct{{[0-9]+}}), 0);
  //CHECK-NEXT:if(sycl::get_pointer_type(result, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result, handle->get_context())!=sycl::usm::alloc::shared) handle->wait();
  //CHECK-NEXT:*result = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  a = cublasIcamax(handle, N, x_C, N, result);
  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  //CHECK-NEXT:mkl::blas::iamax(*handle, N, (std::complex<double>*)x_Z, N, res_temp_ptr_ct{{[0-9]+}});
  //CHECK-NEXT:if(sycl::get_pointer_type(result, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result, handle->get_context())!=sycl::usm::alloc::shared) handle->wait();
  //CHECK-NEXT:*result = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  cublasIzamax(handle, N, x_Z, N, result);

  //CHECK:a = (mkl::blas::rotm(*handle, N, d_C_S, N, d_C_S, N, const_cast<float*>(x_S)), 0);
  a = cublasSrotm(handle, N, d_C_S, N, d_C_S, N, x_S);
  //CHECK:mkl::blas::rotm(*handle, N, d_C_D, N, d_C_D, N, const_cast<double*>(x_D));
  cublasDrotm(handle, N, d_C_D, N, d_C_D, N, x_D);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::copy(*handle, N, x_S, incx, d_C_S, incy), 0);
  a = cublasScopy(handle, N, x_S, incx, d_C_S, incy);
  // CHECK:mkl::blas::copy(*handle, N, x_D, incx, d_C_D, incy);
  cublasDcopy(handle, N, x_D, incx, d_C_D, incy);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::copy(*handle, N, (std::complex<float>*)x_C, incx, (std::complex<float>*)d_C_C, incy), 0);
  a = cublasCcopy(handle, N, x_C, incx, d_C_C, incy);
  // CHECK:mkl::blas::copy(*handle, N, (std::complex<double>*)x_Z, incx, (std::complex<double>*)d_C_Z, incy);
  cublasZcopy(handle, N, x_Z, incx, d_C_Z, incy);


  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::axpy(*handle, N, dpct::get_value(&alpha_S, *handle), x_S, incx, result_S, incy), 0);
  a = cublasSaxpy(handle, N, &alpha_S, x_S, incx, result_S, incy);
  // CHECK:mkl::blas::axpy(*handle, N, dpct::get_value(&alpha_D, *handle), x_D, incx, result_D, incy);
  cublasDaxpy(handle, N, &alpha_D, x_D, incx, result_D, incy);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::axpy(*handle, N, dpct::get_value(&alpha_C, *handle), (std::complex<float>*)x_C, incx, (std::complex<float>*)result_C, incy), 0);
  a = cublasCaxpy(handle, N, &alpha_C, x_C, incx, result_C, incy);
  // CHECK:mkl::blas::axpy(*handle, N, dpct::get_value(&alpha_Z, *handle), (std::complex<double>*)x_Z, incx, (std::complex<double>*)result_Z, incy);
  cublasZaxpy(handle, N, &alpha_Z, x_Z, incx, result_Z, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::scal(*handle, N, dpct::get_value(&alpha_S, *handle), result_S, incx), 0);
  a = cublasSscal(handle, N, &alpha_S, result_S, incx);
  // CHECK:mkl::blas::scal(*handle, N, dpct::get_value(&alpha_D, *handle), result_D, incx);
  cublasDscal(handle, N, &alpha_D, result_D, incx);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::scal(*handle, N, dpct::get_value(&alpha_C, *handle), (std::complex<float>*)result_C, incx), 0);
  a = cublasCscal(handle, N, &alpha_C, result_C, incx);
  // CHECK:mkl::blas::scal(*handle, N, dpct::get_value(&alpha_Z, *handle), (std::complex<double>*)result_Z, incx);
  cublasZscal(handle, N, &alpha_Z, result_Z, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::nrm2(*handle, N, x_S, incx, result_S), 0);
  // CHECK-NEXT:if(sycl::get_pointer_type(result_S, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_context())!=sycl::usm::alloc::shared) handle->wait();
  a = cublasSnrm2(handle, N, x_S, incx, result_S);
  // CHECK:mkl::blas::nrm2(*handle, N, x_D, incx, result_D);
  // CHECK-NEXT:if(sycl::get_pointer_type(result_D, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_context())!=sycl::usm::alloc::shared) handle->wait();
  cublasDnrm2(handle, N, x_D, incx, result_D);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::nrm2(*handle, N, (std::complex<float>*)x_C, incx, result_S), 0);
  // CHECK-NEXT:if(sycl::get_pointer_type(result_S, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_S, handle->get_context())!=sycl::usm::alloc::shared) handle->wait();
  a = cublasScnrm2(handle, N, x_C, incx, result_S);
  // CHECK:mkl::blas::nrm2(*handle, N, (std::complex<double>*)x_Z, incx, result_D);
  // CHECK-NEXT:if(sycl::get_pointer_type(result_D, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(result_D, handle->get_context())!=sycl::usm::alloc::shared) handle->wait();
  cublasDznrm2(handle, N, x_Z, incx, result_D);


  //level 2

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::gemv(*handle, trans2==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans2, N, N, dpct::get_value(&alpha_S, *handle), x_S, lda, y_S, incx, dpct::get_value(&beta_S, *handle), result_S, incy), 0);
  a = cublasSgemv(handle, (cublasOperation_t)trans2, N, N, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  // CHECK:mkl::blas::gemv(*handle, mkl::transpose::nontrans, N, N, dpct::get_value(&alpha_D, *handle), x_D, lda, y_D, incx, dpct::get_value(&beta_D, *handle), result_D, incy);
  cublasDgemv(handle, CUBLAS_OP_N, N, N, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::gemv(*handle, trans2==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans2, N, N, dpct::get_value(&alpha_C, *handle), (std::complex<float>*)x_C, lda, (std::complex<float>*)y_C, incx, dpct::get_value(&beta_C, *handle), (std::complex<float>*)result_C, incy), 0);
  a = cublasCgemv(handle, (cublasOperation_t)trans2, N, N, &alpha_C, x_C, lda, y_C, incx, &beta_C, result_C, incy);
  // CHECK:mkl::blas::gemv(*handle, mkl::transpose::nontrans, N, N, dpct::get_value(&alpha_Z, *handle), (std::complex<double>*)x_Z, lda, (std::complex<double>*)y_Z, incx, dpct::get_value(&beta_Z, *handle), (std::complex<double>*)result_Z, incy);
  cublasZgemv(handle, CUBLAS_OP_N, N, N, &alpha_Z, x_Z, lda, y_Z, incx, &beta_Z, result_Z, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::ger(*handle, N, N, dpct::get_value(&alpha_S, *handle), x_S, incx, y_S, incy, result_S, lda), 0);
  a = cublasSger(handle, N, N, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  // CHECK:mkl::blas::ger(*handle, N, N, dpct::get_value(&alpha_D, *handle), x_D, incx, y_D, incy, result_D, lda);
  cublasDger(handle, N, N, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::geru(*handle, N, N, dpct::get_value(&alpha_C, *handle), (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)result_C, lda), 0);
  a = cublasCgeru(handle, N, N, &alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK:mkl::blas::gerc(*handle, N, N, dpct::get_value(&alpha_C, *handle), (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)result_C, lda);
  cublasCgerc(handle, N, N, &alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::geru(*handle, N, N, dpct::get_value(&alpha_Z, *handle), (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)result_Z, lda), 0);
  a = cublasZgeru(handle, N, N, &alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);
  // CHECK:mkl::blas::gerc(*handle, N, N, dpct::get_value(&alpha_Z, *handle), (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)result_Z, lda);
  cublasZgerc(handle, N, N, &alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);








  //level 3

  __half *d_A_H = 0;
  __half *d_B_H = 0;
  __half *d_C_H = 0;
  __half alpha_H;
  __half beta_H;

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::gemm(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, N, N, N, dpct::get_value(&alpha_S, *handle), d_A_S, N, d_B_S, N, dpct::get_value(&beta_S, *handle), d_C_S, N), 0);
  a = cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  //CHECK:mkl::blas::gemm(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, N, N, N, dpct::get_value(&alpha_D, *handle), d_A_D, N, d_B_D, N, dpct::get_value(&beta_D, *handle), d_C_D, N);
  cublasDgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::gemm(*handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, dpct::get_value(&alpha_C, *handle), (std::complex<float>*)d_A_C, N, (std::complex<float>*)d_B_C, N, dpct::get_value(&beta_C, *handle), (std::complex<float>*)d_C_C, N), 0);
  a = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N);
  //CHECK:mkl::blas::gemm(*handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, dpct::get_value(&alpha_Z, *handle), (std::complex<double>*)d_A_Z, N, (std::complex<double>*)d_B_Z, N, dpct::get_value(&beta_Z, *handle), (std::complex<double>*)d_C_Z, N);
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, &beta_Z, d_C_Z, N);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::gemm_batch(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, N, N, N, dpct::get_value(&alpha_S, *handle), d_A_S, N, 16, d_B_S, N, 16, dpct::get_value(&beta_S, *handle), d_C_S, N, 16, 10), 0);
  a = cublasSgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, 16, d_B_S, N, 16, &beta_S, d_C_S, N, 16, 10);
  //CHECK:mkl::blas::gemm_batch(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, N, N, N, dpct::get_value(&alpha_D, *handle), d_A_D, N, 16, d_B_D, N, 16, dpct::get_value(&beta_D, *handle), d_C_D, N, 16, 10);
  cublasDgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_D, d_A_D, N, 16, d_B_D, N, 16, &beta_D, d_C_D, N, 16, 10);
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::gemm_batch(*handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, dpct::get_value(&alpha_C, *handle), (std::complex<float>*)d_A_C, N, 16, (std::complex<float>*)d_B_C, N, 16, dpct::get_value(&beta_C, *handle), (std::complex<float>*)d_C_C, N, 16, 10), 0);
  a = cublasCgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C, d_A_C, N, 16, d_B_C, N, 16, &beta_C, d_C_C, N, 16, 10);
  //CHECK:mkl::blas::gemm_batch(*handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, dpct::get_value(&alpha_Z, *handle), (std::complex<double>*)d_A_Z, N, 16, (std::complex<double>*)d_B_Z, N, 16, dpct::get_value(&beta_Z, *handle), (std::complex<double>*)d_C_Z, N, 16, 10);
  cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_Z, d_A_Z, N, 16, d_B_Z, N, 16, &beta_Z, d_C_Z, N, 16, 10);
  //CHECK:mkl::blas::gemm_batch(*handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, dpct::get_value(&alpha_H, *handle), d_A_H, N, 16, d_B_H, N, 16, dpct::get_value(&beta_H, *handle), d_C_H, N, 16, 10);
  cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_H, d_A_H, N, 16, d_B_H, N, 16, &beta_H, d_C_H, N, 16, 10);

  cublasOperation_t trans3 = CUBLAS_OP_N;
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::gemm(*handle, trans3, trans3, N, N, N, dpct::get_value(&alpha_H, *handle), d_A_H, N, d_B_H, N, dpct::get_value(&beta_H, *handle), d_C_H, N), 0);
  a = cublasHgemm(handle, trans3, trans3, N, N, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);

  // CHECK: void *alpha, *beta, *A, *B, *C;
  // CHECK-NEXT: int algo = 0;
  void *alpha, *beta, *A, *B, *C;
  cublasGemmAlgo_t algo = CUBLAS_GEMM_ALGO0;
  // CHECK: mkl::blas::gemm(*handle, mkl::transpose::conjtrans, mkl::transpose::conjtrans, N, N, N, dpct::get_value((float*)alpha, *handle), (float*)A, N, (float*)B, N, dpct::get_value((float*)beta, *handle), (float*)C, N);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_32F, N, B, CUDA_R_32F, N, beta, C, CUDA_R_32F, N, CUDA_R_32F, algo);

  float2 alpha_C, beta_C;
  // CHECK: mkl::blas::gemm(*handle, mkl::transpose::conjtrans, mkl::transpose::conjtrans, N, N, N, sycl::vec<float, 1>{dpct::get_value(&alpha_S, *handle)}.convert<sycl::half, sycl::rounding_mode::automatic>()[0], (sycl::half*)A, N, (sycl::half*)B, N, sycl::vec<float, 1>{dpct::get_value(&beta_S, *handle)}.convert<sycl::half, sycl::rounding_mode::automatic>()[0], (sycl::half*)C, N);
  //CHECK-NEXT:mkl::blas::gemm(*handle, mkl::transpose::conjtrans, mkl::transpose::conjtrans, N, N, N, dpct::get_value(&alpha_S, *handle), (sycl::half*)A, N, (sycl::half*)B, N, dpct::get_value(&beta_S, *handle), (float*)C, N);
  //CHECK-NEXT:mkl::blas::gemm(*handle, mkl::transpose::conjtrans, mkl::transpose::conjtrans, N, N, N, dpct::get_value(&alpha_S, *handle), (float*)A, N, (float*)B, N, dpct::get_value(&beta_S, *handle), (float*)C, N);
  //CHECK-NEXT:mkl::blas::gemm(*handle, mkl::transpose::conjtrans, mkl::transpose::conjtrans, N, N, N, dpct::get_value(&alpha_C, *handle), (std::complex<float>*)A, N, (std::complex<float>*)B, N, dpct::get_value(&beta_C, *handle), (std::complex<float>*)C, N);
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

  // CHECK: int64_t m_ct{{[0-9]+}} = N, n_ct{{[0-9]+}} = N, k_ct{{[0-9]+}} = N, lda_ct{{[0-9]+}} = N, ldb_ct{{[0-9]+}} = N, ldc_ct{{[0-9]+}} = N, group_size_ct{{[0-9]+}} = 10;
  // CHECK-NEXT: float alpha_ct{{[0-9]+}} = dpct::get_value(&alpha_S, *handle), beta_ct{{[0-9]+}} = dpct::get_value(&beta_S, *handle);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::gemm_batch(*handle, &trans3, &trans3, &m_ct{{[0-9]+}}, &n_ct{{[0-9]+}}, &k_ct{{[0-9]+}}, &alpha_ct{{[0-9]+}}, d_A_S_array, &lda_ct{{[0-9]+}}, d_B_S_array, &ldb_ct{{[0-9]+}}, &beta_ct{{[0-9]+}}, d_C_S_array, &ldc_ct{{[0-9]+}}, 1, &group_size_ct{{[0-9]+}}, {}), 0);
  // CHECK-NEXT: int64_t m_ct{{[0-9]+}} = N, n_ct{{[0-9]+}} = N, k_ct{{[0-9]+}} = N, lda_ct{{[0-9]+}} = N, ldb_ct{{[0-9]+}} = N, ldc_ct{{[0-9]+}} = N, group_size_ct{{[0-9]+}} = 10;
  // CHECK-NEXT: double alpha_ct{{[0-9]+}} = dpct::get_value(&alpha_D, *handle), beta_ct{{[0-9]+}} = dpct::get_value(&beta_D, *handle);
  // CHECK-NEXT: mkl::blas::gemm_batch(*handle, &trans3, &trans3, &m_ct{{[0-9]+}}, &n_ct{{[0-9]+}}, &k_ct{{[0-9]+}}, &alpha_ct{{[0-9]+}}, d_A_D_array, &lda_ct{{[0-9]+}}, d_B_D_array, &ldb_ct{{[0-9]+}}, &beta_ct{{[0-9]+}}, d_C_D_array, &ldc_ct{{[0-9]+}}, 1, &group_size_ct{{[0-9]+}}, {});
  // CHECK-NEXT: int64_t m_ct{{[0-9]+}} = N, n_ct{{[0-9]+}} = N, k_ct{{[0-9]+}} = N, lda_ct{{[0-9]+}} = N, ldb_ct{{[0-9]+}} = N, ldc_ct{{[0-9]+}} = N, group_size_ct{{[0-9]+}} = 10;
  // CHECK-NEXT: std::complex<float> alpha_ct{{[0-9]+}} = dpct::get_value(&alpha_C, *handle), beta_ct{{[0-9]+}} = dpct::get_value(&beta_C, *handle);
  // CHECK-NEXT: mkl::blas::gemm_batch(*handle, &trans3, &trans3, &m_ct{{[0-9]+}}, &n_ct{{[0-9]+}}, &k_ct{{[0-9]+}}, &alpha_ct{{[0-9]+}}, (const std::complex<float>**)d_A_C_array, &lda_ct{{[0-9]+}}, (const std::complex<float>**)d_B_C_array, &ldb_ct{{[0-9]+}}, &beta_ct{{[0-9]+}}, (std::complex<float>**)d_C_C_array, &ldc_ct{{[0-9]+}}, 1, &group_size_ct{{[0-9]+}}, {});
  // CHECK-NEXT: int64_t m_ct{{[0-9]+}} = N, n_ct{{[0-9]+}} = N, k_ct{{[0-9]+}} = N, lda_ct{{[0-9]+}} = N, ldb_ct{{[0-9]+}} = N, ldc_ct{{[0-9]+}} = N, group_size_ct{{[0-9]+}} = 10;
  // CHECK-NEXT: std::complex<double> alpha_ct{{[0-9]+}} = dpct::get_value(&alpha_Z, *handle), beta_ct{{[0-9]+}} = dpct::get_value(&beta_Z, *handle);
  // CHECK-NEXT: mkl::blas::gemm_batch(*handle, &trans3, &trans3, &m_ct{{[0-9]+}}, &n_ct{{[0-9]+}}, &k_ct{{[0-9]+}}, &alpha_ct{{[0-9]+}}, (const std::complex<double>**)d_A_Z_array, &lda_ct{{[0-9]+}}, (const std::complex<double>**)d_B_Z_array, &ldb_ct{{[0-9]+}}, &beta_ct{{[0-9]+}}, (std::complex<double>**)d_C_Z_array, &ldc_ct{{[0-9]+}}, 1, &group_size_ct{{[0-9]+}}, {});
  a = cublasSgemmBatched(handle, trans3, trans3, N, N, N, &alpha_S, d_A_S_array, N, d_B_S_array, N, &beta_S, d_C_S_array, N, 10);
  cublasDgemmBatched(handle, trans3, trans3, N, N, N, &alpha_D, d_A_D_array, N, d_B_D_array, N, &beta_D, d_C_D_array, N, 10);
  cublasCgemmBatched(handle, trans3, trans3, N, N, N, &alpha_C, d_A_C_array, N, d_B_C_array, N, &beta_C, d_C_C_array, N, 10);
  cublasZgemmBatched(handle, trans3, trans3, N, N, N, &alpha_Z, d_A_Z_array, N, d_B_Z_array, N, &beta_Z, d_C_Z_array, N, 10);

  // CHECK: mkl::side side_ct{{[0-9]+}} = mkl::side::left;
  // CHECK-NEXT: mkl::uplo uplo_ct{{[0-9]+}} = mkl::uplo::lower;
  // CHECK-NEXT: mkl::diag diag_ct{{[0-9]+}} = mkl::diag::unit;
  // CHECK-NEXT: int64_t m_ct{{[0-9]+}} = N, n_ct{{[0-9]+}} = N, lda_ct{{[0-9]+}} = N, ldb_ct{{[0-9]+}} = N, group_size_ct{{[0-9]+}} = 10;
  // CHECK-NEXT: float alpha_ct{{[0-9]+}} = dpct::get_value(&alpha_S, *handle);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::trsm_batch(*handle, &side_ct{{[0-9]+}}, &uplo_ct{{[0-9]+}}, &trans3, &diag_ct{{[0-9]+}}, &m_ct{{[0-9]+}}, &n_ct{{[0-9]+}}, &alpha_ct{{[0-9]+}}, d_A_S_array, &lda_ct{{[0-9]+}}, d_C_S_array, &ldb_ct{{[0-9]+}}, 1, &group_size_ct{{[0-9]+}}, {}), 0);
  // CHECK-NEXT: mkl::side side_ct{{[0-9]+}} = mkl::side::left;
  // CHECK-NEXT: mkl::uplo uplo_ct{{[0-9]+}} = mkl::uplo::lower;
  // CHECK-NEXT: mkl::diag diag_ct{{[0-9]+}} = mkl::diag::unit;
  // CHECK-NEXT: int64_t m_ct{{[0-9]+}} = N, n_ct{{[0-9]+}} = N, lda_ct{{[0-9]+}} = N, ldb_ct{{[0-9]+}} = N, group_size_ct{{[0-9]+}} = 10;
  // CHECK-NEXT: double alpha_ct{{[0-9]+}} = dpct::get_value(&alpha_D, *handle);
  // CHECK-NEXT: mkl::blas::trsm_batch(*handle, &side_ct{{[0-9]+}}, &uplo_ct{{[0-9]+}}, &trans3, &diag_ct{{[0-9]+}}, &m_ct{{[0-9]+}}, &n_ct{{[0-9]+}}, &alpha_ct{{[0-9]+}}, d_A_D_array, &lda_ct{{[0-9]+}}, d_C_D_array, &ldb_ct{{[0-9]+}}, 1, &group_size_ct{{[0-9]+}}, {});
  // CHECK-NEXT: mkl::side side_ct{{[0-9]+}} = mkl::side::left;
  // CHECK-NEXT: mkl::uplo uplo_ct{{[0-9]+}} = mkl::uplo::lower;
  // CHECK-NEXT: mkl::diag diag_ct{{[0-9]+}} = mkl::diag::unit;
  // CHECK-NEXT: int64_t m_ct{{[0-9]+}} = N, n_ct{{[0-9]+}} = N, lda_ct{{[0-9]+}} = N, ldb_ct{{[0-9]+}} = N, group_size_ct{{[0-9]+}} = 10;
  // CHECK-NEXT: std::complex<float> alpha_ct{{[0-9]+}} = dpct::get_value(&alpha_C, *handle);
  // CHECK-NEXT: mkl::blas::trsm_batch(*handle, &side_ct{{[0-9]+}}, &uplo_ct{{[0-9]+}}, &trans3, &diag_ct{{[0-9]+}}, &m_ct{{[0-9]+}}, &n_ct{{[0-9]+}}, &alpha_ct{{[0-9]+}}, (const std::complex<float>**)d_A_C_array, &lda_ct{{[0-9]+}}, (std::complex<float>**)d_C_C_array, &ldb_ct{{[0-9]+}}, 1, &group_size_ct{{[0-9]+}}, {});
  // CHECK-NEXT: mkl::side side_ct{{[0-9]+}} = mkl::side::left;
  // CHECK-NEXT: mkl::uplo uplo_ct{{[0-9]+}} = mkl::uplo::lower;
  // CHECK-NEXT: mkl::diag diag_ct{{[0-9]+}} = mkl::diag::unit;
  // CHECK-NEXT: int64_t m_ct{{[0-9]+}} = N, n_ct{{[0-9]+}} = N, lda_ct{{[0-9]+}} = N, ldb_ct{{[0-9]+}} = N, group_size_ct{{[0-9]+}} = 10;
  // CHECK-NEXT: std::complex<double> alpha_ct{{[0-9]+}} = dpct::get_value(&alpha_Z, *handle);
  // CHECK-NEXT: mkl::blas::trsm_batch(*handle, &side_ct{{[0-9]+}}, &uplo_ct{{[0-9]+}}, &trans3, &diag_ct{{[0-9]+}}, &m_ct{{[0-9]+}}, &n_ct{{[0-9]+}}, &alpha_ct{{[0-9]+}}, (const std::complex<double>**)d_A_Z_array, &lda_ct{{[0-9]+}}, (std::complex<double>**)d_C_Z_array, &ldb_ct{{[0-9]+}}, 1, &group_size_ct{{[0-9]+}}, {});
  a = cublasStrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans3, CUBLAS_DIAG_UNIT, N, N, &alpha_S, d_A_S_array, N, d_C_S_array, N, 10);
  cublasDtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans3, CUBLAS_DIAG_UNIT, N, N, &alpha_D, d_A_D_array, N, d_C_D_array, N, 10);
  cublasCtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans3, CUBLAS_DIAG_UNIT, N, N, &alpha_C, d_A_C_array, N, d_C_C_array, N, 10);
  cublasZtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans3, CUBLAS_DIAG_UNIT, N, N, &alpha_Z, d_A_Z_array, N, d_C_Z_array, N, 10);

  //CHECK:dpct::matrix_mem_copy(d_C_S, d_B_S, N, N, N, N, dpct::device_to_device, *handle);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::trmm(*handle, (mkl::side)side0, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, N, N, dpct::get_value(&alpha_S, *handle), d_A_S, N, d_C_S, N), 0);
  a = cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N);
  //CHECK:dpct::matrix_mem_copy(d_C_D, d_B_D, N, N, N, N, dpct::device_to_device, *handle);
  //CHECK-NEXT:mkl::blas::trmm(*handle, (mkl::side)side0, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, N, N, dpct::get_value(&alpha_D, *handle), d_A_D, N, d_C_D, N);
  cublasDtrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_D, d_A_D, N, d_B_D, N, d_C_D, N);
  //CHECK:dpct::matrix_mem_copy(d_C_C, d_B_C, N, N, N, N, dpct::device_to_device, *handle);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::trmm(*handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, N, N, dpct::get_value(&alpha_C, *handle), (std::complex<float>*)d_A_C, N, (std::complex<float>*)d_C_C, N), 0);
  a = cublasCtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, N, &alpha_C, d_A_C, N, d_B_C, N, d_C_C, N);
  //CHECK:dpct::matrix_mem_copy(d_C_Z, d_B_Z, N, N, N, N, dpct::device_to_device, *handle);
  //CHECK-NEXT:mkl::blas::trmm(*handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, N, N, dpct::get_value(&alpha_Z, *handle), (std::complex<double>*)d_A_Z, N, (std::complex<double>*)d_C_Z, N);
  cublasZtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, d_C_Z, N);


  //CHECK:a = (mkl::blas::gemmt(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, trans1==0 ? mkl::transpose::trans : mkl::transpose::nontrans, N, N, dpct::get_value(&alpha_S, *handle), d_A_S, N, d_B_S, N, dpct::get_value(&beta_S, *handle), d_C_S, N), 0);
  a = cublasSsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans1, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  //CHECK:mkl::blas::gemmt(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, trans1==0 ? mkl::transpose::trans : mkl::transpose::nontrans, N, N, dpct::get_value(&alpha_D, *handle), d_A_D, N, d_B_D, N, dpct::get_value(&beta_D, *handle), d_C_D, N);
  cublasDsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans1, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);



  // CHECK: dpct::matrix_mem_copy(d_C_S, d_B_S, N, N, N, N, dpct::device_to_device, *handle);
  // CHECK-NEXT: mkl::blas::trmm(*handle, (mkl::side)side0, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, N, N, dpct::get_value(&alpha_S, *handle), d_A_S, N, d_C_S, N);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if(int stat = 0){}
  if(int stat = cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N)){}

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if(int stat = (mkl::blas::gemm(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, N, N, N, dpct::get_value(&alpha_S, *handle), d_A_S, N, d_B_S, N, dpct::get_value(&beta_S, *handle), d_C_S, N), 0)){}
  if(int stat = cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){}


}

// CHECK: int foo1() try {
// CHECK-NEXT:   dpct::matrix_mem_copy(d_C_S, d_B_S, N, N, N, N, dpct::device_to_device, *handle);
// CHECK-NEXT:   mkl::blas::trmm(*handle, (mkl::side)side0, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, N, N, dpct::get_value(&alpha_S, *handle), d_A_S, N, d_C_S, N);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a return statement. You may need to rewrite this code.
// CHECK-NEXT:   */
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
int foo1(){
  return cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N);
}

// CHECK:int foo2() try {
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:  */
// CHECK-NEXT:  return (mkl::blas::gemm(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, N, N, N, dpct::get_value(&alpha_S, *handle), d_A_S, N, d_B_S, N, dpct::get_value(&beta_S, *handle), d_C_S, N), 0);
// CHECK-NEXT:}
int foo2(){
  return cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
}
