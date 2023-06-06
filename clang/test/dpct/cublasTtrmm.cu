// RUN: dpct --format-range=none --usm-level=none -out-root %T/cublasTtrmm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasTtrmm/cublasTtrmm.dp.cpp --match-full-lines %s
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
  // CHECK: status = DPCT_CHECK_ERROR(dpct::trmm(*handle, (oneapi::mkl::side)side0, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, m, n, &alpha_S, A_S, lda, B_S, ldb, C_S, ldc));
  // CHECK-NEXT: dpct::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, &alpha_S, A_S, lda, B_S, ldb, C_S, ldc);
  status = cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, &alpha_S, A_S, lda, B_S, ldb, C_S, ldc);
  cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_S, A_S, lda, B_S, ldb, C_S, ldc);

  // CHECK: status = DPCT_CHECK_ERROR(dpct::trmm(*handle, (oneapi::mkl::side)side1, fill1 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans1), (oneapi::mkl::diag)diag1, m, n, &alpha_D, A_D, lda, B_D, ldb, C_D, ldc));
  // CHECK-NEXT: dpct::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, &alpha_D, A_D, lda, B_D, ldb, C_D, ldc);
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


  // CHECK: status = DPCT_CHECK_ERROR(dpct::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::lower, dpct::get_transpose(trans2), oneapi::mkl::diag::nonunit, m, n, &alpha_C, A_C, lda, B_C, ldb, C_C, ldc));
  // CHECK-NEXT: dpct::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, &alpha_C, A_C, lda, B_C, ldb, C_C, ldc);
  status = cublasCtrmm(handle, (cublasSideMode_t)0, (cublasFillMode_t)0, (cublasOperation_t)trans2, (cublasDiagType_t)0, m, n, &alpha_C, A_C, lda, B_C, ldb, C_C, ldc);
  cublasCtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_C, A_C, lda, B_C, ldb, C_C, ldc);


  // CHECK: status = DPCT_CHECK_ERROR(dpct::trmm(*handle, oneapi::mkl::side::right, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, oneapi::mkl::diag::unit, m, n, &alpha_Z, A_Z, lda, B_Z, ldb, C_Z, ldc));
  // CHECK-NEXT: dpct::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, &alpha_Z, A_Z, lda, B_Z, ldb, C_Z, ldc);
  status = cublasZtrmm(handle, (cublasSideMode_t)1, (cublasFillMode_t)1, (cublasOperation_t)2, (cublasDiagType_t)1, m, n, &alpha_Z, A_Z, lda, B_Z, ldb, C_Z, ldc);
  cublasZtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_Z, A_Z, lda, B_Z, ldb, C_Z, ldc);


  // CHECK: dpct::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, dpct::get_transpose(foo1()), oneapi::mkl::diag::nonunit, m, n, &alpha_Z, A_Z, lda, B_Z, ldb, foo2(), ldc);
  cublasZtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, (cublasOperation_t)foo1(), CUBLAS_DIAG_NON_UNIT, m, n, &alpha_Z, A_Z, lda, B_Z, ldb, foo2(), ldc);
}

