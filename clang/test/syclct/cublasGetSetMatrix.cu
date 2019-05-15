// RUN: syclct -extra-arg-before=-std=c++11 -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/cublasGetSetMatrix.sycl.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

constexpr int foo(int i) {
  return i;
}

int main() {
  int rowsA = 100;
  int colsA = 100;
  int lda = 100;
  int ldb = 100;
  float *A = NULL;
  float *d_A = NULL;
  cublasStatus_t status;

#define LDA_MARCO 100
  const int ConstLda = 100;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (syclct::sycl_memcpy((void*)(d_A),(void*)(A),(LDA_MARCO)*(colsA)*(sizeof(A[0])),syclct::host_to_device), 0);
  status = cublasSetMatrix(100, colsA, sizeof(A[0]), A, LDA_MARCO, d_A, 100);

  // CHECK: syclct::sycl_memcpy((void*)(d_A),(void*)(A),(ConstLda)*(colsA)*(sizeof(A[0])),syclct::host_to_device);
  cublasSetMatrix(100, colsA, sizeof(A[0]), A, ConstLda, d_A, 100);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1016:{{[0-9]+}}: Migration of cublasSetMatrix with these parameters is not supported currently, because parameter lda or ldb cannot be evaluated. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, lda, d_A, ldb);
  cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, lda, d_A, ldb);

#define LDB_MARCO 99
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1016:{{[0-9]+}}: Migration of cublasSetMatrix with these parameters is not supported currently, because parameter lda does not equal to ldb. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, 100, d_A, LDB_MARCO);
  cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, 100, d_A, LDB_MARCO);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1018:{{[0-9]+}}: Migration of cublasSetMatrix with these parameters could lead performance issue by auto-migration, because parameter rows cannot be evaluated (rows may be smaller than lda). You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: syclct::sycl_memcpy((void*)(d_A),(void*)(A),(100)*(colsA)*(sizeof(A[0])),syclct::host_to_device);
  cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, 100, d_A, 100);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1018:{{[0-9]+}}: Migration of cublasSetMatrix with these parameters could lead performance issue by auto-migration, because  parameter rows is smaller than lda. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: syclct::sycl_memcpy((void*)(d_A),(void*)(A),(100)*(colsA)*(sizeof(A[0])),syclct::host_to_device);
  cublasSetMatrix(99, colsA, sizeof(A[0]), A, 100, d_A, 100);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1018:{{[0-9]+}}: Migration of cublasSetMatrix with these parameters could lead performance issue by auto-migration, because  parameter rows is smaller than lda. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (syclct::sycl_memcpy((void*)(d_A),(void*)(A),(100)*(colsA)*(sizeof(A[0])),syclct::host_to_device), 0);
  status = cublasSetMatrix(99, colsA, sizeof(A[0]), A, 100, d_A, 100);

  const int ConstLdaNE = lda;
  const int ConstLdbNE = ldb;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1016:{{[0-9]+}}: Migration of cublasGetMatrix with these parameters is not supported currently, because parameter lda or ldb cannot be evaluated. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasGetMatrix(rowsA, colsA, sizeof(A[0]), A, ConstLdaNE, d_A, ConstLdbNE);
  cublasGetMatrix(rowsA, colsA, sizeof(A[0]), A, ConstLdaNE, d_A, ConstLdbNE);

  const int ConstLdaT = 100;
  const int ConstLdbT = 100;
  constexpr int ConstExprLda = 101;
  constexpr int ConstExprLdb = 101;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1016:{{[0-9]+}}: Migration of cublasSetMatrix with these parameters is not supported currently, because parameter lda or ldb cannot be evaluated. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, foo(lda), d_A, foo(ldb));
  cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, foo(lda), d_A, foo(ldb));

  // CHECK: syclct::sycl_memcpy((void*)(d_A),(void*)(A),(foo(ConstLdaT))*(colsA)*(sizeof(A[0])),syclct::host_to_device);
  cublasSetMatrix(100, colsA, sizeof(A[0]), A, foo(ConstLdaT), d_A, foo(ConstLdbT));

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1018:{{[0-9]+}}: Migration of cublasGetMatrix with these parameters could lead performance issue by auto-migration, because  parameter rows is smaller than lda. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: syclct::sycl_memcpy((void*)(d_A),(void*)(A),(foo(ConstExprLda))*(colsA)*(sizeof(A[0])),syclct::device_to_host);
  cublasGetMatrix(100, colsA, sizeof(A[0]), A, foo(ConstExprLda), d_A, ConstExprLdb);

  return 0;
}
