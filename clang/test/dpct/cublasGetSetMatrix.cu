// RUN: c2s --format-range=none --usm-level=none -extra-arg-before=-std=c++14 -out-root %T/cublasGetSetMatrix %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasGetSetMatrix/cublasGetSetMatrix.dp.cpp --match-full-lines %s
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
  // CHECK: sycl::queue *stream;
  cudaStream_t stream;

#define LDA_MARCO 100
  const int ConstLda = 100;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (c2s::matrix_mem_copy((void*)d_A, (void*)A, 100, LDA_MARCO, 100, colsA, sizeof(A[0])), 0);
  status = cublasSetMatrix(100, colsA, sizeof(A[0]), A, LDA_MARCO, d_A, 100);

  // CHECK: c2s::matrix_mem_copy((void*)d_A, (void*)A, 100, ConstLda, 100, colsA, sizeof(A[0]));
  cublasSetMatrix(100, colsA, sizeof(A[0]), A, ConstLda, d_A, 100);

  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasSetMatrix was migrated, but due to parameter(s) lda and/or ldb could not be evaluated, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2s::matrix_mem_copy((void*)d_A, (void*)A, ldb, lda, rowsA, colsA, sizeof(A[0]));
  cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, lda, d_A, ldb);

#define LDB_MARCO 99
  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasSetMatrix was migrated, but due to parameter 100 does not equal to parameter LDB_MARCO, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2s::matrix_mem_copy((void*)d_A, (void*)A, LDB_MARCO, 100, rowsA, colsA, sizeof(A[0]));
  cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, 100, d_A, LDB_MARCO);

  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasSetMatrix was migrated, but due to parameter rowsA could not be evaluated and may be smaller than parameter 100, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2s::matrix_mem_copy((void*)d_A, (void*)A, 100, 100, rowsA, colsA, sizeof(A[0]));
  cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, 100, d_A, 100);

  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasSetMatrix was migrated, but due to parameter 99 is smaller than parameter 100, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2s::matrix_mem_copy((void*)d_A, (void*)A, 100, 100, 99, colsA, sizeof(A[0]));
  cublasSetMatrix(99, colsA, sizeof(A[0]), A, 100, d_A, 100);

  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasSetMatrix was migrated, but due to parameter 99 is smaller than parameter 100, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (c2s::matrix_mem_copy((void*)d_A, (void*)A, 100, 100, 99, colsA, sizeof(A[0])), 0);
  status = cublasSetMatrix(99, colsA, sizeof(A[0]), A, 100, d_A, 100);

  const int ConstLdaNE = lda;
  const int ConstLdbNE = ldb;
  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasGetMatrix was migrated, but due to parameter(s) ConstLdaNE and/or ConstLdbNE could not be evaluated, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2s::matrix_mem_copy((void*)d_A, (void*)A, ConstLdbNE, ConstLdaNE, rowsA, colsA, sizeof(A[0]));
  cublasGetMatrix(rowsA, colsA, sizeof(A[0]), A, ConstLdaNE, d_A, ConstLdbNE);

  const int ConstLdaT = 100;
  const int ConstLdbT = 100;
  constexpr int ConstExprLda = 101;
  constexpr int ConstExprLdb = 101;
  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasSetMatrix was migrated, but due to parameter(s) foo(lda) and/or foo(ldb) could not be evaluated, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2s::matrix_mem_copy((void*)d_A, (void*)A, foo(ldb), foo(lda), rowsA, colsA, sizeof(A[0]));
  cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, foo(lda), d_A, foo(ldb));

  // CHECK: c2s::matrix_mem_copy((void*)d_A, (void*)A, foo(ConstLdbT), foo(ConstLdaT), 100, colsA, sizeof(A[0]));
  cublasSetMatrix(100, colsA, sizeof(A[0]), A, foo(ConstLdaT), d_A, foo(ConstLdbT));

  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasGetMatrix was migrated, but due to parameter 100 is smaller than parameter foo(ConstExprLda), the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2s::matrix_mem_copy((void*)d_A, (void*)A, ConstExprLdb, foo(ConstExprLda), 100, colsA, sizeof(A[0]));
  cublasGetMatrix(100, colsA, sizeof(A[0]), A, foo(ConstExprLda), d_A, ConstExprLdb);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (c2s::matrix_mem_copy((void*)d_A, (void*)A, 100, 100, 100, colsA, sizeof(A[0]), c2s::automatic, *stream, true), 0);
  // CHECK-NEXT: c2s::matrix_mem_copy((void*)d_A, (void*)A, 100, 100, 100, colsA, sizeof(A[0]), c2s::automatic, *stream, true);
  status = cublasSetMatrixAsync(100, colsA, sizeof(A[0]), A, 100, d_A, 100, stream);
  cublasSetMatrixAsync(100, colsA, sizeof(A[0]), A, 100, d_A, 100, stream);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (c2s::matrix_mem_copy((void*)d_A, (void*)A, 100, 100, 100, colsA, sizeof(A[0]), c2s::automatic, *stream, true), 0);
  // CHECK-NEXT: c2s::matrix_mem_copy((void*)d_A, (void*)A, 100, 100, 100, colsA, sizeof(A[0]), c2s::automatic, *stream, true);
  status = cublasGetMatrixAsync(100, colsA, sizeof(A[0]), A, 100, d_A, 100, stream);
  cublasGetMatrixAsync(100, colsA, sizeof(A[0]), A, 100, d_A, 100, stream);

  return 0;
}

