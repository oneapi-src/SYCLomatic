// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --usm-level=none -out-root %T/cublasBatch-9 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasBatch-9/cublasBatch-9.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

void foo1() {
  cublasHandle_t handle;
  int n = 275;
  int nrhs = 275;
  int lda = 275;
  int ldb = 275;
  int ldc = 275;

  float **Barray_S = 0;
  double **Barray_D = 0;
  cuComplex **Barray_C = 0;
  cuDoubleComplex **Barray_Z = 0;

  float **Carray_S = 0;
  double **Carray_D = 0;
  cuComplex **Carray_C = 0;
  cuDoubleComplex **Carray_Z = 0;

  int *PivotArray = 0;
  int *infoArray = 0;
  int batchSize = 10;

  const float *const *Aarray_Sc = 0;
  const double *const *Aarray_Dc = 0;
  const cuComplex *const *Aarray_Cc = 0;
  const cuDoubleComplex *const *Aarray_Zc = 0;

  //CHECK:dpct::getrs_batch_wrapper(*handle, oneapi::mkl::transpose::nontrans, n, nrhs, const_cast<float const **>(Aarray_Sc), lda, PivotArray, Barray_S, ldb, infoArray, batchSize);
  //CHECK-NEXT:dpct::getrs_batch_wrapper(*handle, oneapi::mkl::transpose::nontrans, n, nrhs, const_cast<double const **>(Aarray_Dc), lda, PivotArray, Barray_D, ldb, infoArray, batchSize);
  //CHECK-NEXT:dpct::getrs_batch_wrapper(*handle, oneapi::mkl::transpose::nontrans, n, nrhs, const_cast<sycl::float2 const **>(Aarray_Cc), lda, PivotArray, Barray_C, ldb, infoArray, batchSize);
  //CHECK-NEXT:dpct::getrs_batch_wrapper(*handle, oneapi::mkl::transpose::nontrans, n, nrhs, const_cast<sycl::double2 const **>(Aarray_Zc), lda, PivotArray, Barray_Z, ldb, infoArray, batchSize);
  cublasSgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, Aarray_Sc, lda, PivotArray, Barray_S, ldb, infoArray, batchSize);
  cublasDgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, Aarray_Dc, lda, PivotArray, Barray_D, ldb, infoArray, batchSize);
  cublasCgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, Aarray_Cc, lda, PivotArray, Barray_C, ldb, infoArray, batchSize);
  cublasZgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, Aarray_Zc, lda, PivotArray, Barray_Z, ldb, infoArray, batchSize);

  //CHECK:dpct::getri_batch_wrapper(*handle, n, const_cast<float const **>(Aarray_Sc), lda, PivotArray, Carray_S, ldc, infoArray, batchSize);
  //CHECK-NEXT:dpct::getri_batch_wrapper(*handle, n, const_cast<double const **>(Aarray_Dc), lda, PivotArray, Carray_D, ldc, infoArray, batchSize);
  //CHECK-NEXT:dpct::getri_batch_wrapper(*handle, n, const_cast<sycl::float2 const **>(Aarray_Cc), lda, PivotArray, Carray_C, ldc, infoArray, batchSize);
  //CHECK-NEXT:dpct::getri_batch_wrapper(*handle, n, const_cast<sycl::double2 const **>(Aarray_Zc), lda, PivotArray, Carray_Z, ldc, infoArray, batchSize);
  cublasSgetriBatched(handle, n, Aarray_Sc, lda, PivotArray, Carray_S, ldc, infoArray, batchSize);
  cublasDgetriBatched(handle, n, Aarray_Dc, lda, PivotArray, Carray_D, ldc, infoArray, batchSize);
  cublasCgetriBatched(handle, n, Aarray_Cc, lda, PivotArray, Carray_C, ldc, infoArray, batchSize);
  cublasZgetriBatched(handle, n, Aarray_Zc, lda, PivotArray, Carray_Z, ldc, infoArray, batchSize);
}
