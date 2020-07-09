// RUN: dpct --format-range=none --usm-level=none  -out-root %T %s --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasBatch.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
int main() {
  cublasStatus_t status;
  cublasHandle_t handle;
  int n = 275;
  int m = 275;
  int nrhs = 275;
  int lda = 275;
  int ldb = 275;
  int ldc = 275;

  float **Aarray_S = 0;
  double **Aarray_D = 0;
  cuComplex **Aarray_C = 0;
  cuDoubleComplex **Aarray_Z = 0;

  float **Barray_S = 0;
  double **Barray_D = 0;
  cuComplex **Barray_C = 0;
  cuDoubleComplex **Barray_Z = 0;

  float **Carray_S = 0;
  double **Carray_D = 0;
  cuComplex **Carray_C = 0;
  cuDoubleComplex **Carray_Z = 0;

  float **TauArray_S = 0;
  double **TauArray_D = 0;
  cuComplex **TauArray_C = 0;
  cuDoubleComplex **TauArray_Z = 0;

  int *PivotArray = 0;
  int *infoArray = 0;
  int batchSize = 10;

  int trans0 = 0;
  int trans1 = 1;
  int trans2 = 2;

  const float **Aarray_Sc = 0;
  const double **Aarray_Dc = 0;
  const cuComplex **Aarray_Cc = 0;
  const cuDoubleComplex **Aarray_Zc = 0;

  // getrf_batch
  // CHECK: /*
  // CHECK-NEXT: DPCT1047:{{[0-9]+}}: The meaning of PivotArray in the dpct::getrf_batch_wrapper is different from the cublasSgetrfBatched. You may need to check the migrated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::getrf_batch_wrapper(*handle, n, Aarray_S, lda, PivotArray, infoArray, batchSize), 0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1047:{{[0-9]+}}: The meaning of PivotArray in the dpct::getrf_batch_wrapper is different from the cublasSgetrfBatched. You may need to check the migrated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::getrf_batch_wrapper(*handle, n, Aarray_S, lda, PivotArray, infoArray, batchSize);
  status = cublasSgetrfBatched(handle, n, Aarray_S, lda, PivotArray, infoArray, batchSize);
  cublasSgetrfBatched(handle, n, Aarray_S, lda, PivotArray, infoArray, batchSize);

  // CHECK: /*
  // CHECK-NEXT: DPCT1047:{{[0-9]+}}: The meaning of PivotArray in the dpct::getrf_batch_wrapper is different from the cublasDgetrfBatched. You may need to check the migrated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::getrf_batch_wrapper(*handle, n, Aarray_D, lda, PivotArray, infoArray, batchSize), 0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1047:{{[0-9]+}}: The meaning of PivotArray in the dpct::getrf_batch_wrapper is different from the cublasDgetrfBatched. You may need to check the migrated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::getrf_batch_wrapper(*handle, n, Aarray_D, lda, PivotArray, infoArray, batchSize);
  status = cublasDgetrfBatched(handle, n, Aarray_D, lda, PivotArray, infoArray, batchSize);
  cublasDgetrfBatched(handle, n, Aarray_D, lda, PivotArray, infoArray, batchSize);

  // CHECK: /*
  // CHECK-NEXT: DPCT1047:{{[0-9]+}}: The meaning of PivotArray in the dpct::getrf_batch_wrapper is different from the cublasCgetrfBatched. You may need to check the migrated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::getrf_batch_wrapper(*handle, n, Aarray_C, lda, PivotArray, infoArray, batchSize), 0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1047:{{[0-9]+}}: The meaning of PivotArray in the dpct::getrf_batch_wrapper is different from the cublasCgetrfBatched. You may need to check the migrated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::getrf_batch_wrapper(*handle, n, Aarray_C, lda, PivotArray, infoArray, batchSize);
  status = cublasCgetrfBatched(handle, n, Aarray_C, lda, PivotArray, infoArray, batchSize);
  cublasCgetrfBatched(handle, n, Aarray_C, lda, PivotArray, infoArray, batchSize);

  // CHECK: /*
  // CHECK-NEXT: DPCT1047:{{[0-9]+}}: The meaning of PivotArray in the dpct::getrf_batch_wrapper is different from the cublasZgetrfBatched. You may need to check the migrated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::getrf_batch_wrapper(*handle, n, Aarray_Z, lda, PivotArray, infoArray, batchSize), 0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1047:{{[0-9]+}}: The meaning of PivotArray in the dpct::getrf_batch_wrapper is different from the cublasZgetrfBatched. You may need to check the migrated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::getrf_batch_wrapper(*handle, n, Aarray_Z, lda, PivotArray, infoArray, batchSize);
  status = cublasZgetrfBatched(handle, n, Aarray_Z, lda, PivotArray, infoArray, batchSize);
  cublasZgetrfBatched(handle, n, Aarray_Z, lda, PivotArray, infoArray, batchSize);

  // getrs_batch
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::getrs_batch_wrapper(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, n, nrhs, Aarray_Sc, lda, PivotArray, Barray_S, ldb, infoArray, batchSize), 0);
  // CHECK-NEXT: dpct::getrs_batch_wrapper(*handle, mkl::transpose::nontrans, n, nrhs, Aarray_Sc, lda, PivotArray, Barray_S, ldb, infoArray, batchSize);
  status = cublasSgetrsBatched(handle, (cublasOperation_t)trans0, n, nrhs, Aarray_Sc, lda, PivotArray, Barray_S, ldb, infoArray, batchSize);
  cublasSgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, Aarray_Sc, lda, PivotArray, Barray_S, ldb, infoArray, batchSize);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::getrs_batch_wrapper(*handle, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, n, nrhs, Aarray_Dc, lda, PivotArray, Barray_D, ldb, infoArray, batchSize), 0);
  // CHECK-NEXT: dpct::getrs_batch_wrapper(*handle, mkl::transpose::nontrans, n, nrhs, Aarray_Dc, lda, PivotArray, Barray_D, ldb, infoArray, batchSize);
  status = cublasDgetrsBatched(handle, (cublasOperation_t)trans1, n, nrhs, Aarray_Dc, lda, PivotArray, Barray_D, ldb, infoArray, batchSize);
  cublasDgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, Aarray_Dc, lda, PivotArray, Barray_D, ldb, infoArray, batchSize);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::getrs_batch_wrapper(*handle, trans2==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans2, n, nrhs, Aarray_Cc, lda, PivotArray, Barray_C, ldb, infoArray, batchSize), 0);
  // CHECK-NEXT: dpct::getrs_batch_wrapper(*handle, mkl::transpose::nontrans, n, nrhs, Aarray_Cc, lda, PivotArray, Barray_C, ldb, infoArray, batchSize);
  status = cublasCgetrsBatched(handle, (cublasOperation_t)trans2, n, nrhs, Aarray_Cc, lda, PivotArray, Barray_C, ldb, infoArray, batchSize);
  cublasCgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, Aarray_Cc, lda, PivotArray, Barray_C, ldb, infoArray, batchSize);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::getrs_batch_wrapper(*handle, mkl::transpose::conjtrans, n, nrhs, Aarray_Zc, lda, PivotArray, Barray_Z, ldb, infoArray, batchSize), 0);
  // CHECK-NEXT: dpct::getrs_batch_wrapper(*handle, mkl::transpose::nontrans, n, nrhs, Aarray_Zc, lda, PivotArray, Barray_Z, ldb, infoArray, batchSize);
  status = cublasZgetrsBatched(handle, (cublasOperation_t)2, n, nrhs, Aarray_Zc, lda, PivotArray, Barray_Z, ldb, infoArray, batchSize);
  cublasZgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, Aarray_Zc, lda, PivotArray, Barray_Z, ldb, infoArray, batchSize);

  // getri_batch
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::getri_batch_wrapper(*handle, n, Aarray_Sc, lda, PivotArray, Carray_S, ldc, infoArray, batchSize), 0);
  // CHECK-NEXT: dpct::getri_batch_wrapper(*handle, n, Aarray_Sc, lda, PivotArray, Carray_S, ldc, infoArray, batchSize);
  status = cublasSgetriBatched(handle, n, Aarray_Sc, lda, PivotArray, Carray_S, ldc, infoArray, batchSize);
  cublasSgetriBatched(handle, n, Aarray_Sc, lda, PivotArray, Carray_S, ldc, infoArray, batchSize);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::getri_batch_wrapper(*handle, n, Aarray_Dc, lda, PivotArray, Carray_D, ldc, infoArray, batchSize), 0);
  // CHECK-NEXT: dpct::getri_batch_wrapper(*handle, n, Aarray_Dc, lda, PivotArray, Carray_D, ldc, infoArray, batchSize);
  status = cublasDgetriBatched(handle, n, Aarray_Dc, lda, PivotArray, Carray_D, ldc, infoArray, batchSize);
  cublasDgetriBatched(handle, n, Aarray_Dc, lda, PivotArray, Carray_D, ldc, infoArray, batchSize);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::getri_batch_wrapper(*handle, n, Aarray_Cc, lda, PivotArray, Carray_C, ldc, infoArray, batchSize), 0);
  // CHECK-NEXT: dpct::getri_batch_wrapper(*handle, n, Aarray_Cc, lda, PivotArray, Carray_C, ldc, infoArray, batchSize);
  status = cublasCgetriBatched(handle, n, Aarray_Cc, lda, PivotArray, Carray_C, ldc, infoArray, batchSize);
  cublasCgetriBatched(handle, n, Aarray_Cc, lda, PivotArray, Carray_C, ldc, infoArray, batchSize);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::getri_batch_wrapper(*handle, n, Aarray_Zc, lda, PivotArray, Carray_Z, ldc, infoArray, batchSize), 0);
  // CHECK-NEXT: dpct::getri_batch_wrapper(*handle, n, Aarray_Zc, lda, PivotArray, Carray_Z, ldc, infoArray, batchSize);
  status = cublasZgetriBatched(handle, n, Aarray_Zc, lda, PivotArray, Carray_Z, ldc, infoArray, batchSize);
  cublasZgetriBatched(handle, n, Aarray_Zc, lda, PivotArray, Carray_Z, ldc, infoArray, batchSize);

  // geqrf_batch
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::geqrf_batch_wrapper(*handle, m, n, Aarray_S, lda, TauArray_S, infoArray, batchSize), 0);
  // CHECK-NEXT: dpct::geqrf_batch_wrapper(*handle, m, n, Aarray_S, lda, TauArray_S, infoArray, batchSize);
  status = cublasSgeqrfBatched(handle, m, n, Aarray_S, lda, TauArray_S, infoArray, batchSize);
  cublasSgeqrfBatched(handle, m, n, Aarray_S, lda, TauArray_S, infoArray, batchSize);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::geqrf_batch_wrapper(*handle, m, n, Aarray_D, lda, TauArray_D, infoArray, batchSize), 0);
  // CHECK-NEXT: dpct::geqrf_batch_wrapper(*handle, m, n, Aarray_D, lda, TauArray_D, infoArray, batchSize);
  status = cublasDgeqrfBatched(handle, m, n, Aarray_D, lda, TauArray_D, infoArray, batchSize);
  cublasDgeqrfBatched(handle, m, n, Aarray_D, lda, TauArray_D, infoArray, batchSize);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::geqrf_batch_wrapper(*handle, m, n, Aarray_C, lda, TauArray_C, infoArray, batchSize), 0);
  // CHECK-NEXT: dpct::geqrf_batch_wrapper(*handle, m, n, Aarray_C, lda, TauArray_C, infoArray, batchSize);
  status = cublasCgeqrfBatched(handle, m, n, Aarray_C, lda, TauArray_C, infoArray, batchSize);
  cublasCgeqrfBatched(handle, m, n, Aarray_C, lda, TauArray_C, infoArray, batchSize);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::geqrf_batch_wrapper(*handle, m, n, Aarray_Z, lda, TauArray_Z, infoArray, batchSize), 0);
  // CHECK-NEXT: dpct::geqrf_batch_wrapper(*handle, m, n, Aarray_Z, lda, TauArray_Z, infoArray, batchSize);
  status = cublasZgeqrfBatched(handle, m, n, Aarray_Z, lda, TauArray_Z, infoArray, batchSize);
  cublasZgeqrfBatched(handle, m, n, Aarray_Z, lda, TauArray_Z, infoArray, batchSize);

  return 0;
}
