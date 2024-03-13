// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemmBatched | FileCheck %s -check-prefix=cublasCgemmBatched
// cublasCgemmBatched: CUDA API:
// cublasCgemmBatched-NEXT:   cublasCgemmBatched(
// cublasCgemmBatched-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasCgemmBatched-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasCgemmBatched-NEXT:       alpha /*const cuComplex **/, a /*const cuComplex *const **/, lda /*int*/,
// cublasCgemmBatched-NEXT:       b /*const cuComplex *const **/, ldb /*int*/, beta /*const cuComplex **/,
// cublasCgemmBatched-NEXT:       c /*cuComplex *const **/, ldc /*int*/, group_count /*int*/);
// cublasCgemmBatched-NEXT: Is migrated to:
// cublasCgemmBatched-NEXT:   dpct::gemm_batch(handle->get_queue(), transa, transb, m, n, k, alpha, (const void**)a, dpct::library_data_t::complex_float, lda, (const void**)b, dpct::library_data_t::complex_float, ldb, beta, (void**)c, dpct::library_data_t::complex_float, ldc, group_count, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgeqrfBatched | FileCheck %s -check-prefix=cublasCgeqrfBatched
// cublasCgeqrfBatched: CUDA API:
// cublasCgeqrfBatched-NEXT:   cublasCgeqrfBatched(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
// cublasCgeqrfBatched-NEXT:                       a /*cuComplex *const **/, lda /*int*/,
// cublasCgeqrfBatched-NEXT:                       tau /*cuComplex *const **/, info /*int **/,
// cublasCgeqrfBatched-NEXT:                       group_count /*int*/);
// cublasCgeqrfBatched-NEXT: Is migrated to:
// cublasCgeqrfBatched-NEXT:   dpct::geqrf_batch_wrapper(handle->get_queue(), m, n, const_cast<sycl::float2 **>(a), lda, const_cast<sycl::float2 **>(tau), info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgetrfBatched | FileCheck %s -check-prefix=cublasCgetrfBatched
// cublasCgetrfBatched: CUDA API:
// cublasCgetrfBatched-NEXT:   cublasCgetrfBatched(handle /*cublasHandle_t*/, n /*int*/,
// cublasCgetrfBatched-NEXT:                       a /*cuComplex *const **/, lda /*int*/, ipiv /*int **/,
// cublasCgetrfBatched-NEXT:                       info /*int **/, group_count /*int*/);
// cublasCgetrfBatched-NEXT: Is migrated to:
// cublasCgetrfBatched-NEXT:   dpct::getrf_batch_wrapper(handle->get_queue(), n, const_cast<sycl::float2 **>(a), lda, ipiv, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgetriBatched | FileCheck %s -check-prefix=cublasCgetriBatched
// cublasCgetriBatched: CUDA API:
// cublasCgetriBatched-NEXT:   cublasCgetriBatched(handle /*cublasHandle_t*/, n /*int*/,
// cublasCgetriBatched-NEXT:                       a /*const cuComplex *const **/, lda /*int*/,
// cublasCgetriBatched-NEXT:                       ipiv /*const int **/, c /*cuComplex *const **/,
// cublasCgetriBatched-NEXT:                       ldc /*int*/, info /*int **/, group_count /*int*/);
// cublasCgetriBatched-NEXT: Is migrated to:
// cublasCgetriBatched-NEXT:   dpct::getri_batch_wrapper(handle->get_queue(), n, const_cast<sycl::float2 const **>(a), lda, ipiv, const_cast<sycl::float2 **>(c), ldc, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgetrsBatched | FileCheck %s -check-prefix=cublasCgetrsBatched
// cublasCgetrsBatched: CUDA API:
// cublasCgetrsBatched-NEXT:   cublasCgetrsBatched(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
// cublasCgetrsBatched-NEXT:                       n /*int*/, nrhs /*int*/, a /*const cuComplex *const **/,
// cublasCgetrsBatched-NEXT:                       lda /*int*/, ipiv /*const int **/,
// cublasCgetrsBatched-NEXT:                       b /*cuComplex *const **/, ldb /*int*/, info /*int **/,
// cublasCgetrsBatched-NEXT:                       group_count /*int*/);
// cublasCgetrsBatched-NEXT: Is migrated to:
// cublasCgetrsBatched-NEXT:   dpct::getrs_batch_wrapper(handle->get_queue(), trans, n, nrhs, const_cast<sycl::float2 const **>(a), lda, ipiv, const_cast<sycl::float2 **>(b), ldb, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtrsmBatched | FileCheck %s -check-prefix=cublasCtrsmBatched
// cublasCtrsmBatched: CUDA API:
// cublasCtrsmBatched-NEXT:   cublasCtrsmBatched(
// cublasCtrsmBatched-NEXT:       handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasCtrsmBatched-NEXT:       upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasCtrsmBatched-NEXT:       unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasCtrsmBatched-NEXT:       alpha /*const cuComplex **/, a /*const cuComplex *const **/, lda /*int*/,
// cublasCtrsmBatched-NEXT:       b /*cuComplex *const **/, ldb /*int*/, group_count /*int*/);
// cublasCtrsmBatched-NEXT: Is migrated to:
// cublasCtrsmBatched-NEXT:   dpct::trsm_batch(handle->get_queue(), left_right, upper_lower, transa, unit_diag, m, n, alpha, (const void**)a, dpct::library_data_t::complex_float, lda, (void**)b, dpct::library_data_t::complex_float, ldb, group_count, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgemmBatched | FileCheck %s -check-prefix=cublasDgemmBatched
// cublasDgemmBatched: CUDA API:
// cublasDgemmBatched-NEXT:   cublasDgemmBatched(
// cublasDgemmBatched-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasDgemmBatched-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasDgemmBatched-NEXT:       alpha /*const double **/, a /*const double *const **/, lda /*int*/,
// cublasDgemmBatched-NEXT:       b /*const double *const **/, ldb /*int*/, beta /*const double **/,
// cublasDgemmBatched-NEXT:       c /*double *const **/, ldc /*int*/, group_count /*int*/);
// cublasDgemmBatched-NEXT: Is migrated to:
// cublasDgemmBatched-NEXT:   dpct::gemm_batch(handle->get_queue(), transa, transb, m, n, k, alpha, (const void**)a, dpct::library_data_t::real_double, lda, (const void**)b, dpct::library_data_t::real_double, ldb, beta, (void**)c, dpct::library_data_t::real_double, ldc, group_count, dpct::library_data_t::real_double);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgeqrfBatched | FileCheck %s -check-prefix=cublasDgeqrfBatched
// cublasDgeqrfBatched: CUDA API:
// cublasDgeqrfBatched-NEXT:   cublasDgeqrfBatched(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
// cublasDgeqrfBatched-NEXT:                       a /*double *const **/, lda /*int*/,
// cublasDgeqrfBatched-NEXT:                       tau /*double *const **/, info /*int **/,
// cublasDgeqrfBatched-NEXT:                       group_count /*int*/);
// cublasDgeqrfBatched-NEXT: Is migrated to:
// cublasDgeqrfBatched-NEXT:   dpct::geqrf_batch_wrapper(handle->get_queue(), m, n, const_cast<double **>(a), lda, const_cast<double **>(tau), info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgetrfBatched | FileCheck %s -check-prefix=cublasDgetrfBatched
// cublasDgetrfBatched: CUDA API:
// cublasDgetrfBatched-NEXT:   cublasDgetrfBatched(handle /*cublasHandle_t*/, n /*int*/,
// cublasDgetrfBatched-NEXT:                       a /*double *const **/, lda /*int*/, ipiv /*int **/,
// cublasDgetrfBatched-NEXT:                       info /*int **/, group_count /*int*/);
// cublasDgetrfBatched-NEXT: Is migrated to:
// cublasDgetrfBatched-NEXT:   dpct::getrf_batch_wrapper(handle->get_queue(), n, const_cast<double **>(a), lda, ipiv, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgetriBatched | FileCheck %s -check-prefix=cublasDgetriBatched
// cublasDgetriBatched: CUDA API:
// cublasDgetriBatched-NEXT:   cublasDgetriBatched(handle /*cublasHandle_t*/, n /*int*/,
// cublasDgetriBatched-NEXT:                       a /*const double *const **/, lda /*int*/,
// cublasDgetriBatched-NEXT:                       ipiv /*const int **/, c /*double *const **/, ldc /*int*/,
// cublasDgetriBatched-NEXT:                       info /*int **/, group_count /*int*/);
// cublasDgetriBatched-NEXT: Is migrated to:
// cublasDgetriBatched-NEXT:   dpct::getri_batch_wrapper(handle->get_queue(), n, const_cast<double const **>(a), lda, ipiv, const_cast<double **>(c), ldc, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgetrsBatched | FileCheck %s -check-prefix=cublasDgetrsBatched
// cublasDgetrsBatched: CUDA API:
// cublasDgetrsBatched-NEXT:   cublasDgetrsBatched(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
// cublasDgetrsBatched-NEXT:                       n /*int*/, nrhs /*int*/, a /*const double *const **/,
// cublasDgetrsBatched-NEXT:                       lda /*int*/, ipiv /*const int **/, b /*double *const **/,
// cublasDgetrsBatched-NEXT:                       ldb /*int*/, info /*int **/, group_count /*int*/);
// cublasDgetrsBatched-NEXT: Is migrated to:
// cublasDgetrsBatched-NEXT:   dpct::getrs_batch_wrapper(handle->get_queue(), trans, n, nrhs, const_cast<double const **>(a), lda, ipiv, const_cast<double **>(b), ldb, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtrsmBatched | FileCheck %s -check-prefix=cublasDtrsmBatched
// cublasDtrsmBatched: CUDA API:
// cublasDtrsmBatched-NEXT:   cublasDtrsmBatched(
// cublasDtrsmBatched-NEXT:       handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasDtrsmBatched-NEXT:       upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasDtrsmBatched-NEXT:       unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasDtrsmBatched-NEXT:       alpha /*const double **/, a /*const double *const **/, lda /*int*/,
// cublasDtrsmBatched-NEXT:       b /*double *const **/, ldb /*int*/, group_count /*int*/);
// cublasDtrsmBatched-NEXT: Is migrated to:
// cublasDtrsmBatched-NEXT:   dpct::trsm_batch(handle->get_queue(), left_right, upper_lower, transa, unit_diag, m, n, alpha, (const void**)a, dpct::library_data_t::real_double, lda, (void**)b, dpct::library_data_t::real_double, ldb, group_count, dpct::library_data_t::real_double);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetMathMode | FileCheck %s -check-prefix=cublasGetMathMode
// cublasGetMathMode: CUDA API:
// cublasGetMathMode-NEXT:   cublasGetMathMode(handle /*cublasHandle_t*/, precision /*cublasMath_t **/);
// cublasGetMathMode-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasHgemmBatched | FileCheck %s -check-prefix=cublasHgemmBatched
// cublasHgemmBatched: CUDA API:
// cublasHgemmBatched-NEXT:   cublasHgemmBatched(
// cublasHgemmBatched-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasHgemmBatched-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasHgemmBatched-NEXT:       alpha /*const __half **/, a /*const __half *const **/, lda /*int*/,
// cublasHgemmBatched-NEXT:       b /*const __half *const **/, ldb /*int*/, beta /*const __half **/,
// cublasHgemmBatched-NEXT:       c /*__half *const **/, ldc /*int*/, group_count /*int*/);
// cublasHgemmBatched-NEXT: Is migrated to:
// cublasHgemmBatched-NEXT:   dpct::gemm_batch(handle->get_queue(), transa, transb, m, n, k, alpha, (const void**)a, dpct::library_data_t::real_half, lda, (const void**)b, dpct::library_data_t::real_half, ldb, beta, (void**)c, dpct::library_data_t::real_half, ldc, group_count, dpct::library_data_t::real_half);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetMathMode | FileCheck %s -check-prefix=cublasSetMathMode
// cublasSetMathMode: CUDA API:
// cublasSetMathMode-NEXT:   cublasSetMathMode(handle /*cublasHandle_t*/, precision /*cublasMath_t*/);
// cublasSetMathMode-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgemmBatched | FileCheck %s -check-prefix=cublasSgemmBatched
// cublasSgemmBatched: CUDA API:
// cublasSgemmBatched-NEXT:   cublasSgemmBatched(
// cublasSgemmBatched-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasSgemmBatched-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasSgemmBatched-NEXT:       alpha /*const float **/, a /*const float *const **/, lda /*int*/,
// cublasSgemmBatched-NEXT:       b /*const float *const **/, ldb /*int*/, beta /*const float **/,
// cublasSgemmBatched-NEXT:       c /*float *const **/, ldc /*int*/, group_count /*int*/);
// cublasSgemmBatched-NEXT: Is migrated to:
// cublasSgemmBatched-NEXT:   dpct::gemm_batch(handle->get_queue(), transa, transb, m, n, k, alpha, (const void**)a, dpct::library_data_t::real_float, lda, (const void**)b, dpct::library_data_t::real_float, ldb, beta, (void**)c, dpct::library_data_t::real_float, ldc, group_count, dpct::library_data_t::real_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgeqrfBatched | FileCheck %s -check-prefix=cublasSgeqrfBatched
// cublasSgeqrfBatched: CUDA API:
// cublasSgeqrfBatched-NEXT:   cublasSgeqrfBatched(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
// cublasSgeqrfBatched-NEXT:                       a /*float *const **/, lda /*int*/, tau /*float *const **/,
// cublasSgeqrfBatched-NEXT:                       info /*int **/, group_count /*int*/);
// cublasSgeqrfBatched-NEXT: Is migrated to:
// cublasSgeqrfBatched-NEXT:   dpct::geqrf_batch_wrapper(handle->get_queue(), m, n, const_cast<float **>(a), lda, const_cast<float **>(tau), info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgetrfBatched | FileCheck %s -check-prefix=cublasSgetrfBatched
// cublasSgetrfBatched: CUDA API:
// cublasSgetrfBatched-NEXT:   cublasSgetrfBatched(handle /*cublasHandle_t*/, n /*int*/,
// cublasSgetrfBatched-NEXT:                       a /*float *const **/, lda /*int*/, ipiv /*int **/,
// cublasSgetrfBatched-NEXT:                       info /*int **/, group_count /*int*/);
// cublasSgetrfBatched-NEXT: Is migrated to:
// cublasSgetrfBatched-NEXT:   dpct::getrf_batch_wrapper(handle->get_queue(), n, const_cast<float **>(a), lda, ipiv, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgetriBatched | FileCheck %s -check-prefix=cublasSgetriBatched
// cublasSgetriBatched: CUDA API:
// cublasSgetriBatched-NEXT:   cublasSgetriBatched(handle /*cublasHandle_t*/, n /*int*/,
// cublasSgetriBatched-NEXT:                       a /*const float *const **/, lda /*int*/,
// cublasSgetriBatched-NEXT:                       ipiv /*const int **/, c /*float *const **/, ldc /*int*/,
// cublasSgetriBatched-NEXT:                       info /*int **/, group_count /*int*/);
// cublasSgetriBatched-NEXT: Is migrated to:
// cublasSgetriBatched-NEXT:   dpct::getri_batch_wrapper(handle->get_queue(), n, const_cast<float const **>(a), lda, ipiv, const_cast<float **>(c), ldc, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgetrsBatched | FileCheck %s -check-prefix=cublasSgetrsBatched
// cublasSgetrsBatched: CUDA API:
// cublasSgetrsBatched-NEXT:   cublasSgetrsBatched(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
// cublasSgetrsBatched-NEXT:                       n /*int*/, nrhs /*int*/, a /*const float *const **/,
// cublasSgetrsBatched-NEXT:                       lda /*int*/, ipiv /*const int **/, b /*float *const **/,
// cublasSgetrsBatched-NEXT:                       ldb /*int*/, info /*int **/, group_count /*int*/);
// cublasSgetrsBatched-NEXT: Is migrated to:
// cublasSgetrsBatched-NEXT:   dpct::getrs_batch_wrapper(handle->get_queue(), trans, n, nrhs, const_cast<float const **>(a), lda, ipiv, const_cast<float **>(b), ldb, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStrsmBatched | FileCheck %s -check-prefix=cublasStrsmBatched
// cublasStrsmBatched: CUDA API:
// cublasStrsmBatched-NEXT:   cublasStrsmBatched(
// cublasStrsmBatched-NEXT:       handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasStrsmBatched-NEXT:       upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasStrsmBatched-NEXT:       unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasStrsmBatched-NEXT:       alpha /*const float **/, a /*const float *const **/, lda /*int*/,
// cublasStrsmBatched-NEXT:       b /*float *const **/, ldb /*int*/, group_count /*int*/);
// cublasStrsmBatched-NEXT: Is migrated to:
// cublasStrsmBatched-NEXT:   dpct::trsm_batch(handle->get_queue(), left_right, upper_lower, transa, unit_diag, m, n, alpha, (const void**)a, dpct::library_data_t::real_float, lda, (void**)b, dpct::library_data_t::real_float, ldb, group_count, dpct::library_data_t::real_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgemmBatched | FileCheck %s -check-prefix=cublasZgemmBatched
// cublasZgemmBatched: CUDA API:
// cublasZgemmBatched-NEXT:   cublasZgemmBatched(
// cublasZgemmBatched-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasZgemmBatched-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasZgemmBatched-NEXT:       alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex *const **/,
// cublasZgemmBatched-NEXT:       lda /*int*/, b /*const cuDoubleComplex *const **/, ldb /*int*/,
// cublasZgemmBatched-NEXT:       beta /*const cuDoubleComplex **/, c /*cuDoubleComplex *const **/,
// cublasZgemmBatched-NEXT:       ldc /*int*/, group_count /*int*/);
// cublasZgemmBatched-NEXT: Is migrated to:
// cublasZgemmBatched-NEXT:   dpct::gemm_batch(handle->get_queue(), transa, transb, m, n, k, alpha, (const void**)a, dpct::library_data_t::complex_double, lda, (const void**)b, dpct::library_data_t::complex_double, ldb, beta, (void**)c, dpct::library_data_t::complex_double, ldc, group_count, dpct::library_data_t::complex_double);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgeqrfBatched | FileCheck %s -check-prefix=cublasZgeqrfBatched
// cublasZgeqrfBatched: CUDA API:
// cublasZgeqrfBatched-NEXT:   cublasZgeqrfBatched(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
// cublasZgeqrfBatched-NEXT:                       a /*cuDoubleComplex *const **/, lda /*int*/,
// cublasZgeqrfBatched-NEXT:                       tau /*cuDoubleComplex *const **/, info /*int **/,
// cublasZgeqrfBatched-NEXT:                       group_count /*int*/);
// cublasZgeqrfBatched-NEXT: Is migrated to:
// cublasZgeqrfBatched-NEXT:   dpct::geqrf_batch_wrapper(handle->get_queue(), m, n, const_cast<sycl::double2 **>(a), lda, const_cast<sycl::double2 **>(tau), info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgetrfBatched | FileCheck %s -check-prefix=cublasZgetrfBatched
// cublasZgetrfBatched: CUDA API:
// cublasZgetrfBatched-NEXT:   cublasZgetrfBatched(handle /*cublasHandle_t*/, n /*int*/,
// cublasZgetrfBatched-NEXT:                       a /*cuDoubleComplex *const **/, lda /*int*/,
// cublasZgetrfBatched-NEXT:                       ipiv /*int **/, info /*int **/, group_count /*int*/);
// cublasZgetrfBatched-NEXT: Is migrated to:
// cublasZgetrfBatched-NEXT:   dpct::getrf_batch_wrapper(handle->get_queue(), n, const_cast<sycl::double2 **>(a), lda, ipiv, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgetriBatched | FileCheck %s -check-prefix=cublasZgetriBatched
// cublasZgetriBatched: CUDA API:
// cublasZgetriBatched-NEXT:   cublasZgetriBatched(handle /*cublasHandle_t*/, n /*int*/,
// cublasZgetriBatched-NEXT:                       a /*const cuDoubleComplex *const **/, lda /*int*/,
// cublasZgetriBatched-NEXT:                       ipiv /*const int **/, c /*cuDoubleComplex *const **/,
// cublasZgetriBatched-NEXT:                       ldc /*int*/, info /*int **/, group_count /*int*/);
// cublasZgetriBatched-NEXT: Is migrated to:
// cublasZgetriBatched-NEXT:   dpct::getri_batch_wrapper(handle->get_queue(), n, const_cast<sycl::double2 const **>(a), lda, ipiv, const_cast<sycl::double2 **>(c), ldc, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgetrsBatched | FileCheck %s -check-prefix=cublasZgetrsBatched
// cublasZgetrsBatched: CUDA API:
// cublasZgetrsBatched-NEXT:   cublasZgetrsBatched(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
// cublasZgetrsBatched-NEXT:                       n /*int*/, nrhs /*int*/,
// cublasZgetrsBatched-NEXT:                       a /*const cuDoubleComplex *const **/, lda /*int*/,
// cublasZgetrsBatched-NEXT:                       ipiv /*const int **/, b /*cuDoubleComplex *const **/,
// cublasZgetrsBatched-NEXT:                       ldb /*int*/, info /*int **/, group_count /*int*/);
// cublasZgetrsBatched-NEXT: Is migrated to:
// cublasZgetrsBatched-NEXT:   dpct::getrs_batch_wrapper(handle->get_queue(), trans, n, nrhs, const_cast<sycl::double2 const **>(a), lda, ipiv, const_cast<sycl::double2 **>(b), ldb, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtrsmBatched | FileCheck %s -check-prefix=cublasZtrsmBatched
// cublasZtrsmBatched: CUDA API:
// cublasZtrsmBatched-NEXT:   cublasZtrsmBatched(
// cublasZtrsmBatched-NEXT:       handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZtrsmBatched-NEXT:       upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasZtrsmBatched-NEXT:       unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasZtrsmBatched-NEXT:       alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex *const **/,
// cublasZtrsmBatched-NEXT:       lda /*int*/, b /*cuDoubleComplex *const **/, ldb /*int*/,
// cublasZtrsmBatched-NEXT:       group_count /*int*/);
// cublasZtrsmBatched-NEXT: Is migrated to:
// cublasZtrsmBatched-NEXT:   dpct::trsm_batch(handle->get_queue(), left_right, upper_lower, transa, unit_diag, m, n, alpha, (const void**)a, dpct::library_data_t::complex_double, lda, (void**)b, dpct::library_data_t::complex_double, ldb, group_count, dpct::library_data_t::complex_double);
