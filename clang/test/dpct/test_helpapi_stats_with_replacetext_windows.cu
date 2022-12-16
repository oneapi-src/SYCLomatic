// UNSUPPORTED: -linux-
// RUN: mkdir %T/test_helpapi_stats_with_replacetext_windows
// RUN: cd %T/test_helpapi_stats_with_replacetext_windows
// RUN: dpct --format-range=none -out-root %T/test_helpapi_stats_with_replacetext_windows %s --cuda-include-path="%cuda-path/include"  --report-type=stats -- -x cuda --cuda-host-only > stats.txt
// RUN: echo "// CHECK: File name, LOC migrated to SYCL, LOC migrated to helper functions, LOC not needed to migrate, LOC not able to migrate" > %T/test_helpapi_stats_with_replacetext_windows/test_helpapi_stats_with_replacetext_ref.txt
// RUN: echo "// CHECK-NEXT: {{(.+)}}\test_helpapi_stats_with_replacetext_windows.cu, 13, 17, 47, 0" >> %T/test_helpapi_stats_with_replacetext_windows/test_helpapi_stats_with_replacetext_ref.txt
// RUN: FileCheck --match-full-lines --input-file %T/test_helpapi_stats_with_replacetext_windows/stats.txt %T/test_helpapi_stats_with_replacetext_windows/test_helpapi_stats_with_replacetext_ref.txt
// RUN: cd ..
// RUN: rm -rf ./test_helpapi_stats_with_replacetext_windows

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

  //The following 16 APIs are migrated to helper functions. The number "16" is this test focuses on.
  cublasSgetrfBatched(handle, n, Aarray_S, lda, PivotArray, infoArray, batchSize);
  cublasDgetrfBatched(handle, n, Aarray_D, lda, PivotArray, infoArray, batchSize);
  cublasCgetrfBatched(handle, n, Aarray_C, lda, PivotArray, infoArray, batchSize);
  cublasZgetrfBatched(handle, n, Aarray_Z, lda, PivotArray, infoArray, batchSize);
  cublasSgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, Aarray_Sc, lda, PivotArray, Barray_S, ldb, infoArray, batchSize);
  cublasDgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, Aarray_Dc, lda, PivotArray, Barray_D, ldb, infoArray, batchSize);
  cublasCgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, Aarray_Cc, lda, PivotArray, Barray_C, ldb, infoArray, batchSize);
  cublasZgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, Aarray_Zc, lda, PivotArray, Barray_Z, ldb, infoArray, batchSize);
  cublasSgetriBatched(handle, n, Aarray_Sc, lda, PivotArray, Carray_S, ldc, infoArray, batchSize);
  cublasDgetriBatched(handle, n, Aarray_Dc, lda, PivotArray, Carray_D, ldc, infoArray, batchSize);
  cublasCgetriBatched(handle, n, Aarray_Cc, lda, PivotArray, Carray_C, ldc, infoArray, batchSize);
  cublasZgetriBatched(handle, n, Aarray_Zc, lda, PivotArray, Carray_Z, ldc, infoArray, batchSize);
  cublasSgeqrfBatched(handle, m, n, Aarray_S, lda, TauArray_S, infoArray, batchSize);
  cublasDgeqrfBatched(handle, m, n, Aarray_D, lda, TauArray_D, infoArray, batchSize);
  cublasCgeqrfBatched(handle, m, n, Aarray_C, lda, TauArray_C, infoArray, batchSize);
  cublasZgeqrfBatched(handle, m, n, Aarray_Z, lda, TauArray_Z, infoArray, batchSize);

  return 0;
}
