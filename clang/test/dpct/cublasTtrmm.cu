// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasTtrmm.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int foo1();
cuDoubleComplex * foo2();


int main(){
  cublasStatus_t status;
  cublasHandle_t handle;
  int n = 275;
  int m = 275;
  int lda = 275;
  int ldb = 275;
  int ldc = 275;
  const float *A_S = 0;
  const float *B_S = 0;
  float *C_S = 0;
  float alpha_S = 1.0f;
  const double *A_D = 0;
  const double *B_D = 0;
  double *C_D = 0;
  double alpha_D = 1.0;

  int side0 = 0; int side1 = 1; int fill0 = 0; int fill1 = 1;
  int trans0 = 0; int trans1 = 1; int trans2 = 2; int diag0 = 0; int diag1 = 1;
  // CHECK: {
  // CHECK-NEXT: dpct::matrix_mem_copy(C_S, B_S, ldc, ldb, m, n, dpct::device_to_device, *handle);
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (oneapi::mkl::blas::trmm(*handle, (oneapi::mkl::side)side0, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, (oneapi::mkl::diag)diag0, m, n, dpct::get_value(&alpha_S, *handle), A_S_buf_ct{{[0-9]+}}, lda, C_S_buf_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: dpct::matrix_mem_copy(C_S, B_S, ldc, ldb, m, n, dpct::device_to_device, *handle);
  // CHECK-NEXT: auto A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_S);
  // CHECK-NEXT: oneapi::mkl::blas::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, dpct::get_value(&alpha_S, *handle), A_S_buf_ct{{[0-9]+}}, lda, C_S_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, &alpha_S, A_S, lda, B_S, ldb, C_S, ldc);
  cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_S, A_S, lda, B_S, ldb, C_S, ldc);


  // CHECK: {
  // CHECK-NEXT: dpct::matrix_mem_copy(C_D, B_D, ldc, ldb, m, n, dpct::device_to_device, *handle);
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (oneapi::mkl::blas::trmm(*handle, (oneapi::mkl::side)side1, fill1==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans1==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans1, (oneapi::mkl::diag)diag1, m, n, dpct::get_value(&alpha_D, *handle), A_D_buf_ct{{[0-9]+}}, lda, C_D_buf_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: dpct::matrix_mem_copy(C_D, B_D, ldc, ldb, m, n, dpct::device_to_device, *handle);
  // CHECK-NEXT: auto A_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto C_D_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_D);
  // CHECK-NEXT: oneapi::mkl::blas::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, dpct::get_value(&alpha_D, *handle), A_D_buf_ct{{[0-9]+}}, lda, C_D_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasDtrmm(handle, (cublasSideMode_t)side1, (cublasFillMode_t)fill1, (cublasOperation_t)trans1, (cublasDiagType_t)diag1, m, n, &alpha_D, A_D, lda, B_D, ldb, C_D, ldc);
  cublasDtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_D, A_D, lda, B_D, ldb, C_D, ldc);


  const cuComplex *A_C = 0;
  const cuComplex *B_C = 0;
  cuComplex *C_C = 0;
  cuComplex alpha_C = make_cuComplex(1.0f,0.0f);
  const cuDoubleComplex *A_Z = 0;
  const cuDoubleComplex *B_Z = 0;
  cuDoubleComplex *C_Z = 0;
  cuDoubleComplex alpha_Z = make_cuDoubleComplex(1.0,0.0);


  // CHECK: {
  // CHECK-NEXT: dpct::matrix_mem_copy(C_C, B_C, ldc, ldb, m, n, dpct::device_to_device, *handle);
  // CHECK-NEXT: auto A_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (oneapi::mkl::blas::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::lower, trans2==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans2, oneapi::mkl::diag::nonunit, m, n, dpct::get_value(&alpha_C, *handle), A_C_buf_ct{{[0-9]+}}, lda, C_C_buf_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: dpct::matrix_mem_copy(C_C, B_C, ldc, ldb, m, n, dpct::device_to_device, *handle);
  // CHECK-NEXT: auto A_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto C_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(C_C);
  // CHECK-NEXT: oneapi::mkl::blas::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, dpct::get_value(&alpha_C, *handle), A_C_buf_ct{{[0-9]+}}, lda, C_C_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasCtrmm(handle, (cublasSideMode_t)0, (cublasFillMode_t)0, (cublasOperation_t)trans2, (cublasDiagType_t)0, m, n, &alpha_C, A_C, lda, B_C, ldb, C_C, ldc);
  cublasCtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_C, A_C, lda, B_C, ldb, C_C, ldc);


  // CHECK: {
  // CHECK-NEXT: dpct::matrix_mem_copy(C_Z, B_Z, ldc, ldb, m, n, dpct::device_to_device, *handle);
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:3: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (oneapi::mkl::blas::trmm(*handle, oneapi::mkl::side::right, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, oneapi::mkl::diag::unit, m, n, dpct::get_value(&alpha_Z, *handle), A_Z_buf_ct{{[0-9]+}}, lda, C_Z_buf_ct{{[0-9]+}}, ldc), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: dpct::matrix_mem_copy(C_Z, B_Z, ldc, ldb, m, n, dpct::device_to_device, *handle);
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto C_Z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(C_Z);
  // CHECK-NEXT: oneapi::mkl::blas::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, dpct::get_value(&alpha_Z, *handle), A_Z_buf_ct{{[0-9]+}}, lda, C_Z_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  status = cublasZtrmm(handle, (cublasSideMode_t)1, (cublasFillMode_t)1, (cublasOperation_t)2, (cublasDiagType_t)1, m, n, &alpha_Z, A_Z, lda, B_Z, ldb, C_Z, ldc);
  cublasZtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_Z, A_Z, lda, B_Z, ldb, C_Z, ldc);


  // CHECK: {
  // CHECK-NEXT: auto foo2_ptr_ct{{[0-9]+}} = foo2();
  // CHECK-NEXT: dpct::matrix_mem_copy(foo2_ptr_ct{{[0-9]+}}, B_Z, ldc, ldb, m, n, dpct::device_to_device, *handle);
  // CHECK-NEXT: auto transpose_ct{{[0-9]+}} = foo1();
  // CHECK-NEXT: auto A_Z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto foo2_ptr_ct{{[0-9]+}}_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(foo2_ptr_ct{{[0-9]+}});
  // CHECK-NEXT: oneapi::mkl::blas::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, (int)transpose_ct{{[0-9]+}}==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)transpose_ct{{[0-9]+}}, oneapi::mkl::diag::nonunit, m, n, dpct::get_value(&alpha_Z, *handle), A_Z_buf_ct{{[0-9]+}}, lda,  foo2_ptr_ct{{[0-9]+}}_buf_ct{{[0-9]+}}, ldc);
  // CHECK-NEXT: }
  cublasZtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, (cublasOperation_t)foo1(), CUBLAS_DIAG_NON_UNIT, m, n, &alpha_Z, A_Z, lda, B_Z, ldb, foo2(), ldc);
}
