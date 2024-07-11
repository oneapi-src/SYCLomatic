// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1
// UNSUPPORTED: v8.0, v9.0, v9.1

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSpotrsBatched | FileCheck %s -check-prefix=cusolverDnSpotrsBatched
// cusolverDnSpotrsBatched: CUDA API:
// cusolverDnSpotrsBatched-NEXT:   cusolverDnSpotrsBatched(
// cusolverDnSpotrsBatched-NEXT:       handle /*cusolverDnHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cusolverDnSpotrsBatched-NEXT:       n /*int*/, nrhs /*int*/, a /*float ***/, lda /*int*/, b /*float ***/,
// cusolverDnSpotrsBatched-NEXT:       ldb /*int*/, info /*int **/, group_count /*int*/);
// cusolverDnSpotrsBatched-NEXT: Is migrated to:
// cusolverDnSpotrsBatched-NEXT:   dpct::lapack::potrs_batch(*handle, upper_lower, n, nrhs, a, lda, b, ldb, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSpotrfBatched | FileCheck %s -check-prefix=cusolverDnSpotrfBatched
// cusolverDnSpotrfBatched: CUDA API:
// cusolverDnSpotrfBatched-NEXT:   cusolverDnSpotrfBatched(handle /*cusolverDnHandle_t*/,
// cusolverDnSpotrfBatched-NEXT:                           upper_lower /*cublasFillMode_t*/, n /*int*/,
// cusolverDnSpotrfBatched-NEXT:                           a /*float ***/, lda /*int*/, info /*int **/,
// cusolverDnSpotrfBatched-NEXT:                           group_count /*int*/);
// cusolverDnSpotrfBatched-NEXT: Is migrated to:
// cusolverDnSpotrfBatched-NEXT:   dpct::lapack::potrf_batch(*handle, upper_lower, n, a, lda, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDpotrfBatched | FileCheck %s -check-prefix=cusolverDnDpotrfBatched
// cusolverDnDpotrfBatched: CUDA API:
// cusolverDnDpotrfBatched-NEXT:   cusolverDnDpotrfBatched(handle /*cusolverDnHandle_t*/,
// cusolverDnDpotrfBatched-NEXT:                           upper_lower /*cublasFillMode_t*/, n /*int*/,
// cusolverDnDpotrfBatched-NEXT:                           a /*double ***/, lda /*int*/, info /*int **/,
// cusolverDnDpotrfBatched-NEXT:                           group_count /*int*/);
// cusolverDnDpotrfBatched-NEXT: Is migrated to:
// cusolverDnDpotrfBatched-NEXT:   dpct::lapack::potrf_batch(*handle, upper_lower, n, a, lda, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCpotrsBatched | FileCheck %s -check-prefix=cusolverDnCpotrsBatched
// cusolverDnCpotrsBatched: CUDA API:
// cusolverDnCpotrsBatched-NEXT:   cusolverDnCpotrsBatched(
// cusolverDnCpotrsBatched-NEXT:       handle /*cusolverDnHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cusolverDnCpotrsBatched-NEXT:       n /*int*/, nrhs /*int*/, a /*cuComplex ***/, lda /*int*/,
// cusolverDnCpotrsBatched-NEXT:       b /*cuComplex ***/, ldb /*int*/, info /*int **/, group_count /*int*/);
// cusolverDnCpotrsBatched-NEXT: Is migrated to:
// cusolverDnCpotrsBatched-NEXT:   dpct::lapack::potrs_batch(*handle, upper_lower, n, nrhs, a, lda, b, ldb, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZpotrsBatched | FileCheck %s -check-prefix=cusolverDnZpotrsBatched
// cusolverDnZpotrsBatched: CUDA API:
// cusolverDnZpotrsBatched-NEXT:   cusolverDnZpotrsBatched(handle /*cusolverDnHandle_t*/,
// cusolverDnZpotrsBatched-NEXT:                           upper_lower /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZpotrsBatched-NEXT:                           nrhs /*int*/, a /*cuDoubleComplex ***/, lda /*int*/,
// cusolverDnZpotrsBatched-NEXT:                           b /*cuDoubleComplex ***/, ldb /*int*/, info /*int **/,
// cusolverDnZpotrsBatched-NEXT:                           group_count /*int*/);
// cusolverDnZpotrsBatched-NEXT: Is migrated to:
// cusolverDnZpotrsBatched-NEXT:   dpct::lapack::potrs_batch(*handle, upper_lower, n, nrhs, a, lda, b, ldb, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDpotrsBatched | FileCheck %s -check-prefix=cusolverDnDpotrsBatched
// cusolverDnDpotrsBatched: CUDA API:
// cusolverDnDpotrsBatched-NEXT:   cusolverDnDpotrsBatched(
// cusolverDnDpotrsBatched-NEXT:       handle /*cusolverDnHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cusolverDnDpotrsBatched-NEXT:       n /*int*/, nrhs /*int*/, a /*double ***/, lda /*int*/, b /*double ***/,
// cusolverDnDpotrsBatched-NEXT:       ldb /*int*/, info /*int **/, group_count /*int*/);
// cusolverDnDpotrsBatched-NEXT: Is migrated to:
// cusolverDnDpotrsBatched-NEXT:   dpct::lapack::potrs_batch(*handle, upper_lower, n, nrhs, a, lda, b, ldb, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZpotrfBatched | FileCheck %s -check-prefix=cusolverDnZpotrfBatched
// cusolverDnZpotrfBatched: CUDA API:
// cusolverDnZpotrfBatched-NEXT:   cusolverDnZpotrfBatched(handle /*cusolverDnHandle_t*/,
// cusolverDnZpotrfBatched-NEXT:                           upper_lower /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZpotrfBatched-NEXT:                           a /*cuDoubleComplex ***/, lda /*int*/, info /*int **/,
// cusolverDnZpotrfBatched-NEXT:                           group_count /*int*/);
// cusolverDnZpotrfBatched-NEXT: Is migrated to:
// cusolverDnZpotrfBatched-NEXT:   dpct::lapack::potrf_batch(*handle, upper_lower, n, a, lda, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCpotrfBatched | FileCheck %s -check-prefix=cusolverDnCpotrfBatched
// cusolverDnCpotrfBatched: CUDA API:
// cusolverDnCpotrfBatched-NEXT:   cusolverDnCpotrfBatched(handle /*cusolverDnHandle_t*/,
// cusolverDnCpotrfBatched-NEXT:                           upper_lower /*cublasFillMode_t*/, n /*int*/,
// cusolverDnCpotrfBatched-NEXT:                           a /*cuComplex ***/, lda /*int*/, info /*int **/,
// cusolverDnCpotrfBatched-NEXT:                           group_count /*int*/);
// cusolverDnCpotrfBatched-NEXT: Is migrated to:
// cusolverDnCpotrfBatched-NEXT:   dpct::lapack::potrf_batch(*handle, upper_lower, n, a, lda, info, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCreate | FileCheck %s -check-prefix=cusolverDnCreate
// cusolverDnCreate: CUDA API:
// cusolverDnCreate-NEXT:   cusolverDnHandle_t handle;
// cusolverDnCreate-NEXT:   cusolverDnCreate(&handle /*cusolverDnHandle_t **/);
// cusolverDnCreate-NEXT: Is migrated to:
// cusolverDnCreate-NEXT:   dpct::queue_ptr handle;
// cusolverDnCreate-NEXT:   handle = &dpct::get_in_order_queue();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDestroy | FileCheck %s -check-prefix=cusolverDnDestroy
// cusolverDnDestroy: CUDA API:
// cusolverDnDestroy-NEXT:   cusolverDnDestroy(handle /*cusolverDnHandle_t*/);
// cusolverDnDestroy-NEXT: Is migrated to:
// cusolverDnDestroy-NEXT:   handle = nullptr;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSetStream | FileCheck %s -check-prefix=cusolverDnSetStream
// cusolverDnSetStream: CUDA API:
// cusolverDnSetStream-NEXT:   cusolverDnSetStream(handle /*cusolverDnHandle_t*/, s /*cudaStream_t*/);
// cusolverDnSetStream-NEXT: Is migrated to:
// cusolverDnSetStream-NEXT:   handle = s;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnGetStream | FileCheck %s -check-prefix=cusolverDnGetStream
// cusolverDnGetStream: CUDA API:
// cusolverDnGetStream-NEXT:   cudaStream_t s;
// cusolverDnGetStream-NEXT:   cusolverDnGetStream(handle /*cusolverDnHandle_t*/, &s /*cudaStream_t **/);
// cusolverDnGetStream-NEXT: Is migrated to:
// cusolverDnGetStream-NEXT:   dpct::queue_ptr s;
// cusolverDnGetStream-NEXT:   s = handle;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCreateSyevjInfo | FileCheck %s -check-prefix=cusolverDnCreateSyevjInfo
// cusolverDnCreateSyevjInfo: CUDA API:
// cusolverDnCreateSyevjInfo-NEXT:   cusolverDnCreateSyevjInfo(info /*syevjInfo_t **/);
// cusolverDnCreateSyevjInfo-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDestroySyevjInfo | FileCheck %s -check-prefix=cusolverDnDestroySyevjInfo
// cusolverDnDestroySyevjInfo: CUDA API:
// cusolverDnDestroySyevjInfo-NEXT:   cusolverDnDestroySyevjInfo(info /*syevjInfo_t*/);
// cusolverDnDestroySyevjInfo-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCreateGesvdjInfo | FileCheck %s -check-prefix=cusolverDnCreateGesvdjInfo
// cusolverDnCreateGesvdjInfo: CUDA API:
// cusolverDnCreateGesvdjInfo-NEXT:   cusolverDnCreateGesvdjInfo(info /*gesvdjInfo_t **/);
// cusolverDnCreateGesvdjInfo-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDestroyGesvdjInfo | FileCheck %s -check-prefix=cusolverDnDestroyGesvdjInfo
// cusolverDnDestroyGesvdjInfo: CUDA API:
// cusolverDnDestroyGesvdjInfo-NEXT:   cusolverDnDestroyGesvdjInfo(info /*gesvdjInfo_t*/);
// cusolverDnDestroyGesvdjInfo-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSpotrf_bufferSize | FileCheck %s -check-prefix=cusolverDnSpotrf_bufferSize
// cusolverDnSpotrf_bufferSize: CUDA API:
// cusolverDnSpotrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnSpotrf_bufferSize-NEXT:   cusolverDnSpotrf_bufferSize(
// cusolverDnSpotrf_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnSpotrf_bufferSize-NEXT:       a /*float **/, lda /*int*/, &buffer_size /*int **/);
// cusolverDnSpotrf_bufferSize-NEXT: Is migrated to:
// cusolverDnSpotrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnSpotrf_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::potrf_scratchpad_size<float>(
// cusolverDnSpotrf_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*float **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDpotrf_bufferSize | FileCheck %s -check-prefix=cusolverDnDpotrf_bufferSize
// cusolverDnDpotrf_bufferSize: CUDA API:
// cusolverDnDpotrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnDpotrf_bufferSize-NEXT:   cusolverDnDpotrf_bufferSize(
// cusolverDnDpotrf_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnDpotrf_bufferSize-NEXT:       a /*double **/, lda /*int*/, &buffer_size /*int **/);
// cusolverDnDpotrf_bufferSize-NEXT: Is migrated to:
// cusolverDnDpotrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnDpotrf_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::potrf_scratchpad_size<double>(
// cusolverDnDpotrf_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*double **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCpotrf_bufferSize | FileCheck %s -check-prefix=cusolverDnCpotrf_bufferSize
// cusolverDnCpotrf_bufferSize: CUDA API:
// cusolverDnCpotrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnCpotrf_bufferSize-NEXT:   cusolverDnCpotrf_bufferSize(
// cusolverDnCpotrf_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnCpotrf_bufferSize-NEXT:       a /*cuComplex **/, lda /*int*/, &buffer_size /*int **/);
// cusolverDnCpotrf_bufferSize-NEXT: Is migrated to:
// cusolverDnCpotrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnCpotrf_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<float>>(
// cusolverDnCpotrf_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*cuComplex **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZpotrf_bufferSize | FileCheck %s -check-prefix=cusolverDnZpotrf_bufferSize
// cusolverDnZpotrf_bufferSize: CUDA API:
// cusolverDnZpotrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnZpotrf_bufferSize-NEXT:   cusolverDnZpotrf_bufferSize(
// cusolverDnZpotrf_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZpotrf_bufferSize-NEXT:       a /*cuDoubleComplex **/, lda /*int*/, &buffer_size /*int **/);
// cusolverDnZpotrf_bufferSize-NEXT: Is migrated to:
// cusolverDnZpotrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnZpotrf_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<double>>(
// cusolverDnZpotrf_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*cuDoubleComplex **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSpotrf | FileCheck %s -check-prefix=cusolverDnSpotrf
// cusolverDnSpotrf: CUDA API:
// cusolverDnSpotrf-NEXT:   cusolverDnSpotrf(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSpotrf-NEXT:                    n /*int*/, a /*float **/, lda /*int*/, buffer /*float **/,
// cusolverDnSpotrf-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnSpotrf-NEXT: Is migrated to:
// cusolverDnSpotrf-NEXT:   oneapi::mkl::lapack::potrf(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSpotrf-NEXT:                    n /*int*/, (float*)a /*float **/, lda /*int*/, (float*)buffer /*float **/,
// cusolverDnSpotrf-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDpotrf | FileCheck %s -check-prefix=cusolverDnDpotrf
// cusolverDnDpotrf: CUDA API:
// cusolverDnDpotrf-NEXT:   cusolverDnDpotrf(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDpotrf-NEXT:                    n /*int*/, a /*double **/, lda /*int*/, buffer /*double **/,
// cusolverDnDpotrf-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnDpotrf-NEXT: Is migrated to:
// cusolverDnDpotrf-NEXT:   oneapi::mkl::lapack::potrf(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDpotrf-NEXT:                    n /*int*/, (double*)a /*double **/, lda /*int*/, (double*)buffer /*double **/,
// cusolverDnDpotrf-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCpotrf | FileCheck %s -check-prefix=cusolverDnCpotrf
// cusolverDnCpotrf: CUDA API:
// cusolverDnCpotrf-NEXT:   cusolverDnCpotrf(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnCpotrf-NEXT:                    n /*int*/, a /*cuComplex **/, lda /*int*/,
// cusolverDnCpotrf-NEXT:                    buffer /*cuComplex **/, buffer_size /*int*/, info /*int **/);
// cusolverDnCpotrf-NEXT: Is migrated to:
// cusolverDnCpotrf-NEXT:   oneapi::mkl::lapack::potrf(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnCpotrf-NEXT:                    n /*int*/, (std::complex<float>*)a /*cuComplex **/, lda /*int*/,
// cusolverDnCpotrf-NEXT:                    (std::complex<float>*)buffer /*cuComplex **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZpotrf | FileCheck %s -check-prefix=cusolverDnZpotrf
// cusolverDnZpotrf: CUDA API:
// cusolverDnZpotrf-NEXT:   cusolverDnZpotrf(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnZpotrf-NEXT:                    n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZpotrf-NEXT:                    buffer /*cuDoubleComplex **/, buffer_size /*int*/,
// cusolverDnZpotrf-NEXT:                    info /*int **/);
// cusolverDnZpotrf-NEXT: Is migrated to:
// cusolverDnZpotrf-NEXT:   oneapi::mkl::lapack::potrf(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnZpotrf-NEXT:                    n /*int*/, (std::complex<double>*)a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZpotrf-NEXT:                    (std::complex<double>*)buffer /*cuDoubleComplex **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSpotrs | FileCheck %s -check-prefix=cusolverDnSpotrs
// cusolverDnSpotrs: CUDA API:
// cusolverDnSpotrs-NEXT:   cusolverDnSpotrs(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSpotrs-NEXT:                    n /*int*/, nrhs /*int*/, a /*const float **/, lda /*int*/,
// cusolverDnSpotrs-NEXT:                    b /*float **/, ldb /*int*/, info /*int **/);
// cusolverDnSpotrs-NEXT: Is migrated to:
// cusolverDnSpotrs-NEXT:   {
// cusolverDnSpotrs-NEXT:   std::int64_t scratchpad_size_ct1 = oneapi::mkl::lapack::potrs_scratchpad_size<float>(*handle ,uplo ,n ,nrhs ,lda ,ldb);
// cusolverDnSpotrs-NEXT:   float *scratchpad_ct2 = sycl::malloc_device<float>(scratchpad_size_ct1, *handle);
// cusolverDnSpotrs-NEXT:   sycl::event event_ct3;
// cusolverDnSpotrs-NEXT:   event_ct3 = oneapi::mkl::lapack::potrs(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSpotrs-NEXT:                    n /*int*/, nrhs /*int*/, (float*)a /*const float **/, lda /*int*/,
// cusolverDnSpotrs-NEXT:                    (float*)b /*float **/, ldb, scratchpad_ct2, scratchpad_size_ct1 /*int **/);
// cusolverDnSpotrs-NEXT:   std::vector<void *> ws_vec_ct4{scratchpad_ct2};
// cusolverDnSpotrs-NEXT:   dpct::async_dpct_free(ws_vec_ct4, {event_ct3}, *handle);
// cusolverDnSpotrs-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDpotrs | FileCheck %s -check-prefix=cusolverDnDpotrs
// cusolverDnDpotrs: CUDA API:
// cusolverDnDpotrs-NEXT:   cusolverDnDpotrs(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDpotrs-NEXT:                    n /*int*/, nrhs /*int*/, a /*const double **/, lda /*int*/,
// cusolverDnDpotrs-NEXT:                    b /*double **/, ldb /*int*/, info /*int **/);
// cusolverDnDpotrs-NEXT: Is migrated to:
// cusolverDnDpotrs-NEXT:   {
// cusolverDnDpotrs-NEXT:   std::int64_t scratchpad_size_ct1 = oneapi::mkl::lapack::potrs_scratchpad_size<double>(*handle ,uplo ,n ,nrhs ,lda ,ldb);
// cusolverDnDpotrs-NEXT:   double *scratchpad_ct2 = sycl::malloc_device<double>(scratchpad_size_ct1, *handle);
// cusolverDnDpotrs-NEXT:   sycl::event event_ct3;
// cusolverDnDpotrs-NEXT:   event_ct3 = oneapi::mkl::lapack::potrs(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDpotrs-NEXT:                    n /*int*/, nrhs /*int*/, (double*)a /*const double **/, lda /*int*/,
// cusolverDnDpotrs-NEXT:                    (double*)b /*double **/, ldb, scratchpad_ct2, scratchpad_size_ct1 /*int **/);
// cusolverDnDpotrs-NEXT:   std::vector<void *> ws_vec_ct4{scratchpad_ct2};
// cusolverDnDpotrs-NEXT:   dpct::async_dpct_free(ws_vec_ct4, {event_ct3}, *handle);
// cusolverDnDpotrs-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCpotrs | FileCheck %s -check-prefix=cusolverDnCpotrs
// cusolverDnCpotrs: CUDA API:
// cusolverDnCpotrs-NEXT:   cusolverDnCpotrs(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnCpotrs-NEXT:                    n /*int*/, nrhs /*int*/, a /*const cuComplex **/,
// cusolverDnCpotrs-NEXT:                    lda /*int*/, b /*cuComplex **/, ldb /*int*/, info /*int **/);
// cusolverDnCpotrs-NEXT: Is migrated to:
// cusolverDnCpotrs-NEXT:   {
// cusolverDnCpotrs-NEXT:   std::int64_t scratchpad_size_ct1 = oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<float>>(*handle ,uplo ,n ,nrhs ,lda ,ldb);
// cusolverDnCpotrs-NEXT:   std::complex<float> *scratchpad_ct2 = sycl::malloc_device<std::complex<float>>(scratchpad_size_ct1, *handle);
// cusolverDnCpotrs-NEXT:   sycl::event event_ct3;
// cusolverDnCpotrs-NEXT:   event_ct3 = oneapi::mkl::lapack::potrs(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnCpotrs-NEXT:                    n /*int*/, nrhs /*int*/, (std::complex<float>*)a /*const cuComplex **/,
// cusolverDnCpotrs-NEXT:                    lda /*int*/, (std::complex<float>*)b /*cuComplex **/, ldb, scratchpad_ct2, scratchpad_size_ct1 /*int **/);
// cusolverDnCpotrs-NEXT:   std::vector<void *> ws_vec_ct4{scratchpad_ct2};
// cusolverDnCpotrs-NEXT:   dpct::async_dpct_free(ws_vec_ct4, {event_ct3}, *handle);
// cusolverDnCpotrs-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZpotrs | FileCheck %s -check-prefix=cusolverDnZpotrs
// cusolverDnZpotrs: CUDA API:
// cusolverDnZpotrs-NEXT:   cusolverDnZpotrs(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnZpotrs-NEXT:                    n /*int*/, nrhs /*int*/, a /*const cuDoubleComplex **/,
// cusolverDnZpotrs-NEXT:                    lda /*int*/, b /*cuDoubleComplex **/, ldb /*int*/,
// cusolverDnZpotrs-NEXT:                    info /*int **/);
// cusolverDnZpotrs-NEXT: Is migrated to:
// cusolverDnZpotrs-NEXT:   {
// cusolverDnZpotrs-NEXT:   std::int64_t scratchpad_size_ct1 = oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<double>>(*handle ,uplo ,n ,nrhs ,lda ,ldb);
// cusolverDnZpotrs-NEXT:   std::complex<double> *scratchpad_ct2 = sycl::malloc_device<std::complex<double>>(scratchpad_size_ct1, *handle);
// cusolverDnZpotrs-NEXT:   sycl::event event_ct3;
// cusolverDnZpotrs-NEXT:   event_ct3 = oneapi::mkl::lapack::potrs(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnZpotrs-NEXT:                    n /*int*/, nrhs /*int*/, (std::complex<double>*)a /*const cuDoubleComplex **/,
// cusolverDnZpotrs-NEXT:                    lda /*int*/, (std::complex<double>*)b /*cuDoubleComplex **/, ldb, scratchpad_ct2, scratchpad_size_ct1 /*int **/);
// cusolverDnZpotrs-NEXT:   std::vector<void *> ws_vec_ct4{scratchpad_ct2};
// cusolverDnZpotrs-NEXT:   dpct::async_dpct_free(ws_vec_ct4, {event_ct3}, *handle);
// cusolverDnZpotrs-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSgetrf_bufferSize | FileCheck %s -check-prefix=cusolverDnSgetrf_bufferSize
// cusolverDnSgetrf_bufferSize: CUDA API:
// cusolverDnSgetrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnSgetrf_bufferSize-NEXT:   cusolverDnSgetrf_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnSgetrf_bufferSize-NEXT:                               n /*int*/, a /*float **/, lda /*int*/,
// cusolverDnSgetrf_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnSgetrf_bufferSize-NEXT: Is migrated to:
// cusolverDnSgetrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnSgetrf_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::getrf_scratchpad_size<float>(*handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnSgetrf_bufferSize-NEXT:                               n /*float **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDgetrf_bufferSize | FileCheck %s -check-prefix=cusolverDnDgetrf_bufferSize
// cusolverDnDgetrf_bufferSize: CUDA API:
// cusolverDnDgetrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnDgetrf_bufferSize-NEXT:   cusolverDnDgetrf_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnDgetrf_bufferSize-NEXT:                               n /*int*/, a /*double **/, lda /*int*/,
// cusolverDnDgetrf_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnDgetrf_bufferSize-NEXT: Is migrated to:
// cusolverDnDgetrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnDgetrf_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::getrf_scratchpad_size<double>(*handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnDgetrf_bufferSize-NEXT:                               n /*double **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCgetrf_bufferSize | FileCheck %s -check-prefix=cusolverDnCgetrf_bufferSize
// cusolverDnCgetrf_bufferSize: CUDA API:
// cusolverDnCgetrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnCgetrf_bufferSize-NEXT:   cusolverDnCgetrf_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnCgetrf_bufferSize-NEXT:                               n /*int*/, a /*cuComplex **/, lda /*int*/,
// cusolverDnCgetrf_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnCgetrf_bufferSize-NEXT: Is migrated to:
// cusolverDnCgetrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnCgetrf_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<float>>(*handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnCgetrf_bufferSize-NEXT:                               n /*cuComplex **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZgetrf_bufferSize | FileCheck %s -check-prefix=cusolverDnZgetrf_bufferSize
// cusolverDnZgetrf_bufferSize: CUDA API:
// cusolverDnZgetrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnZgetrf_bufferSize-NEXT:   cusolverDnZgetrf_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnZgetrf_bufferSize-NEXT:                               n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZgetrf_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnZgetrf_bufferSize-NEXT: Is migrated to:
// cusolverDnZgetrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnZgetrf_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<double>>(*handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnZgetrf_bufferSize-NEXT:                               n /*cuDoubleComplex **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSgetrf | FileCheck %s -check-prefix=cusolverDnSgetrf
// cusolverDnSgetrf: CUDA API:
// cusolverDnSgetrf-NEXT:   cusolverDnSgetrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnSgetrf-NEXT:                    a /*float **/, lda /*int*/, buffer /*float **/,
// cusolverDnSgetrf-NEXT:                    ipiv /*int **/, info /*int **/);
// cusolverDnSgetrf-NEXT: Is migrated to:
// cusolverDnSgetrf-NEXT:   {
// cusolverDnSgetrf-NEXT:   int64_t result_temp_pointer6;
// cusolverDnSgetrf-NEXT:   std::int64_t scratchpad_size_ct1 = oneapi::mkl::lapack::getrf_scratchpad_size<float>(*handle ,m ,n ,lda);
// cusolverDnSgetrf-NEXT:   oneapi::mkl::lapack::getrf(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnSgetrf-NEXT:                    (float*)a /*float **/, lda, &result_temp_pointer6 /*int*/, (float*)buffer, scratchpad_size_ct1 /*int **/);
// cusolverDnSgetrf-NEXT:    *ipiv = result_temp_pointer6;
// cusolverDnSgetrf-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDgetrf | FileCheck %s -check-prefix=cusolverDnDgetrf
// cusolverDnDgetrf: CUDA API:
// cusolverDnDgetrf-NEXT:   cusolverDnDgetrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnDgetrf-NEXT:                    a /*double **/, lda /*int*/, buffer /*double **/,
// cusolverDnDgetrf-NEXT:                    ipiv /*int **/, info /*int **/);
// cusolverDnDgetrf-NEXT: Is migrated to:
// cusolverDnDgetrf-NEXT:   {
// cusolverDnDgetrf-NEXT:   int64_t result_temp_pointer6;
// cusolverDnDgetrf-NEXT:   std::int64_t scratchpad_size_ct1 = oneapi::mkl::lapack::getrf_scratchpad_size<double>(*handle ,m ,n ,lda);
// cusolverDnDgetrf-NEXT:   oneapi::mkl::lapack::getrf(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnDgetrf-NEXT:                    (double*)a /*double **/, lda, &result_temp_pointer6 /*int*/, (double*)buffer, scratchpad_size_ct1 /*int **/);
// cusolverDnDgetrf-NEXT:    *ipiv = result_temp_pointer6;
// cusolverDnDgetrf-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCgetrf | FileCheck %s -check-prefix=cusolverDnCgetrf
// cusolverDnCgetrf: CUDA API:
// cusolverDnCgetrf-NEXT:   cusolverDnCgetrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnCgetrf-NEXT:                    a /*cuComplex **/, lda /*int*/, buffer /*cuComplex **/,
// cusolverDnCgetrf-NEXT:                    ipiv /*int **/, info /*int **/);
// cusolverDnCgetrf-NEXT: Is migrated to:
// cusolverDnCgetrf-NEXT:   {
// cusolverDnCgetrf-NEXT:   int64_t result_temp_pointer6;
// cusolverDnCgetrf-NEXT:   std::int64_t scratchpad_size_ct1 = oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<float>>(*handle ,m ,n ,lda);
// cusolverDnCgetrf-NEXT:   oneapi::mkl::lapack::getrf(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnCgetrf-NEXT:                    (std::complex<float>*)a /*cuComplex **/, lda, &result_temp_pointer6 /*int*/, (std::complex<float>*)buffer, scratchpad_size_ct1 /*int **/);
// cusolverDnCgetrf-NEXT:    *ipiv = result_temp_pointer6;
// cusolverDnCgetrf-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZgetrf | FileCheck %s -check-prefix=cusolverDnZgetrf
// cusolverDnZgetrf: CUDA API:
// cusolverDnZgetrf-NEXT:   cusolverDnZgetrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnZgetrf-NEXT:                    a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZgetrf-NEXT:                    buffer /*cuDoubleComplex **/, ipiv /*int **/,
// cusolverDnZgetrf-NEXT:                    info /*int **/);
// cusolverDnZgetrf-NEXT: Is migrated to:
// cusolverDnZgetrf-NEXT:   {
// cusolverDnZgetrf-NEXT:   int64_t result_temp_pointer6;
// cusolverDnZgetrf-NEXT:   std::int64_t scratchpad_size_ct1 = oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<double>>(*handle ,m ,n ,lda);
// cusolverDnZgetrf-NEXT:   oneapi::mkl::lapack::getrf(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnZgetrf-NEXT:                    (std::complex<double>*)a /*cuDoubleComplex **/, lda, &result_temp_pointer6 /*int*/,
// cusolverDnZgetrf-NEXT:                    (std::complex<double>*)buffer, scratchpad_size_ct1 /*int **/);
// cusolverDnZgetrf-NEXT:    *ipiv = result_temp_pointer6;
// cusolverDnZgetrf-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSgetrs | FileCheck %s -check-prefix=cusolverDnSgetrs
// cusolverDnSgetrs: CUDA API:
// cusolverDnSgetrs-NEXT:   cusolverDnSgetrs(handle /*cusolverDnHandle_t*/, trans /*cublasOperation_t*/,
// cusolverDnSgetrs-NEXT:                    n /*int*/, nrhs /*int*/, a /*const float **/, lda /*int*/,
// cusolverDnSgetrs-NEXT:                    ipiv /*const int **/, b /*float **/, ldb /*int*/,
// cusolverDnSgetrs-NEXT:                    info /*int **/);
// cusolverDnSgetrs-NEXT: Is migrated to:
// cusolverDnSgetrs-NEXT:   {
// cusolverDnSgetrs-NEXT:   int64_t result_temp_pointer6;
// cusolverDnSgetrs-NEXT:   std::int64_t scratchpad_size_ct1 = oneapi::mkl::lapack::getrs_scratchpad_size<float>(*handle ,trans ,n ,nrhs ,lda ,ldb);
// cusolverDnSgetrs-NEXT:   float *scratchpad_ct2 = sycl::malloc_device<float>(scratchpad_size_ct1, *handle);
// cusolverDnSgetrs-NEXT:   sycl::event event_ct3;
// cusolverDnSgetrs-NEXT:   event_ct3 = oneapi::mkl::lapack::getrs(*handle /*cusolverDnHandle_t*/, trans /*cublasOperation_t*/,
// cusolverDnSgetrs-NEXT:                    n /*int*/, nrhs /*int*/, (float*)a /*const float **/, lda /*int*/,
// cusolverDnSgetrs-NEXT:                    &result_temp_pointer6 /*const int **/, (float*)b /*float **/, ldb, scratchpad_ct2, scratchpad_size_ct1 /*int **/);
// cusolverDnSgetrs-NEXT:    *ipiv = result_temp_pointer6;
// cusolverDnSgetrs-NEXT:   std::vector<void *> ws_vec_ct4{scratchpad_ct2};
// cusolverDnSgetrs-NEXT:   dpct::async_dpct_free(ws_vec_ct4, {event_ct3}, *handle);
// cusolverDnSgetrs-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDgetrs | FileCheck %s -check-prefix=cusolverDnDgetrs
// cusolverDnDgetrs: CUDA API:
// cusolverDnDgetrs-NEXT:   cusolverDnDgetrs(handle /*cusolverDnHandle_t*/, trans /*cublasOperation_t*/,
// cusolverDnDgetrs-NEXT:                    n /*int*/, nrhs /*int*/, a /*const double **/, lda /*int*/,
// cusolverDnDgetrs-NEXT:                    ipiv /*const int **/, b /*double **/, ldb /*int*/,
// cusolverDnDgetrs-NEXT:                    info /*int **/);
// cusolverDnDgetrs-NEXT: Is migrated to:
// cusolverDnDgetrs-NEXT:   {
// cusolverDnDgetrs-NEXT:   int64_t result_temp_pointer6;
// cusolverDnDgetrs-NEXT:   std::int64_t scratchpad_size_ct1 = oneapi::mkl::lapack::getrs_scratchpad_size<double>(*handle ,trans ,n ,nrhs ,lda ,ldb);
// cusolverDnDgetrs-NEXT:   double *scratchpad_ct2 = sycl::malloc_device<double>(scratchpad_size_ct1, *handle);
// cusolverDnDgetrs-NEXT:   sycl::event event_ct3;
// cusolverDnDgetrs-NEXT:   event_ct3 = oneapi::mkl::lapack::getrs(*handle /*cusolverDnHandle_t*/, trans /*cublasOperation_t*/,
// cusolverDnDgetrs-NEXT:                    n /*int*/, nrhs /*int*/, (double*)a /*const double **/, lda /*int*/,
// cusolverDnDgetrs-NEXT:                    &result_temp_pointer6 /*const int **/, (double*)b /*double **/, ldb, scratchpad_ct2, scratchpad_size_ct1 /*int **/);
// cusolverDnDgetrs-NEXT:    *ipiv = result_temp_pointer6;
// cusolverDnDgetrs-NEXT:   std::vector<void *> ws_vec_ct4{scratchpad_ct2};
// cusolverDnDgetrs-NEXT:   dpct::async_dpct_free(ws_vec_ct4, {event_ct3}, *handle);
// cusolverDnDgetrs-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCgetrs | FileCheck %s -check-prefix=cusolverDnCgetrs
// cusolverDnCgetrs: CUDA API:
// cusolverDnCgetrs-NEXT:   cusolverDnCgetrs(handle /*cusolverDnHandle_t*/, trans /*cublasOperation_t*/,
// cusolverDnCgetrs-NEXT:                    n /*int*/, nrhs /*int*/, a /*const cuComplex **/,
// cusolverDnCgetrs-NEXT:                    lda /*int*/, ipiv /*const int **/, b /*cuComplex **/,
// cusolverDnCgetrs-NEXT:                    ldb /*int*/, info /*int **/);
// cusolverDnCgetrs-NEXT: Is migrated to:
// cusolverDnCgetrs-NEXT:   {
// cusolverDnCgetrs-NEXT:   int64_t result_temp_pointer6;
// cusolverDnCgetrs-NEXT:   std::int64_t scratchpad_size_ct1 = oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<float>>(*handle ,trans ,n ,nrhs ,lda ,ldb);
// cusolverDnCgetrs-NEXT:   std::complex<float> *scratchpad_ct2 = sycl::malloc_device<std::complex<float>>(scratchpad_size_ct1, *handle);
// cusolverDnCgetrs-NEXT:   sycl::event event_ct3;
// cusolverDnCgetrs-NEXT:   event_ct3 = oneapi::mkl::lapack::getrs(*handle /*cusolverDnHandle_t*/, trans /*cublasOperation_t*/,
// cusolverDnCgetrs-NEXT:                    n /*int*/, nrhs /*int*/, (std::complex<float>*)a /*const cuComplex **/,
// cusolverDnCgetrs-NEXT:                    lda /*int*/, &result_temp_pointer6 /*const int **/, (std::complex<float>*)b /*cuComplex **/,
// cusolverDnCgetrs-NEXT:                    ldb, scratchpad_ct2, scratchpad_size_ct1 /*int **/);
// cusolverDnCgetrs-NEXT:    *ipiv = result_temp_pointer6;
// cusolverDnCgetrs-NEXT:   std::vector<void *> ws_vec_ct4{scratchpad_ct2};
// cusolverDnCgetrs-NEXT:   dpct::async_dpct_free(ws_vec_ct4, {event_ct3}, *handle);
// cusolverDnCgetrs-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZgetrs | FileCheck %s -check-prefix=cusolverDnZgetrs
// cusolverDnZgetrs: CUDA API:
// cusolverDnZgetrs-NEXT:   cusolverDnZgetrs(handle /*cusolverDnHandle_t*/, trans /*cublasOperation_t*/,
// cusolverDnZgetrs-NEXT:                    n /*int*/, nrhs /*int*/, a /*const cuDoubleComplex **/,
// cusolverDnZgetrs-NEXT:                    lda /*int*/, ipiv /*const int **/, b /*cuDoubleComplex **/,
// cusolverDnZgetrs-NEXT:                    ldb /*int*/, info /*int **/);
// cusolverDnZgetrs-NEXT: Is migrated to:
// cusolverDnZgetrs-NEXT:   {
// cusolverDnZgetrs-NEXT:   int64_t result_temp_pointer6;
// cusolverDnZgetrs-NEXT:   std::int64_t scratchpad_size_ct1 = oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<double>>(*handle ,trans ,n ,nrhs ,lda ,ldb);
// cusolverDnZgetrs-NEXT:   std::complex<double> *scratchpad_ct2 = sycl::malloc_device<std::complex<double>>(scratchpad_size_ct1, *handle);
// cusolverDnZgetrs-NEXT:   sycl::event event_ct3;
// cusolverDnZgetrs-NEXT:   event_ct3 = oneapi::mkl::lapack::getrs(*handle /*cusolverDnHandle_t*/, trans /*cublasOperation_t*/,
// cusolverDnZgetrs-NEXT:                    n /*int*/, nrhs /*int*/, (std::complex<double>*)a /*const cuDoubleComplex **/,
// cusolverDnZgetrs-NEXT:                    lda /*int*/, &result_temp_pointer6 /*const int **/, (std::complex<double>*)b /*cuDoubleComplex **/,
// cusolverDnZgetrs-NEXT:                    ldb, scratchpad_ct2, scratchpad_size_ct1 /*int **/);
// cusolverDnZgetrs-NEXT:    *ipiv = result_temp_pointer6;
// cusolverDnZgetrs-NEXT:   std::vector<void *> ws_vec_ct4{scratchpad_ct2};
// cusolverDnZgetrs-NEXT:   dpct::async_dpct_free(ws_vec_ct4, {event_ct3}, *handle);
// cusolverDnZgetrs-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSgeqrf_bufferSize | FileCheck %s -check-prefix=cusolverDnSgeqrf_bufferSize
// cusolverDnSgeqrf_bufferSize: CUDA API:
// cusolverDnSgeqrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnSgeqrf_bufferSize-NEXT:   cusolverDnSgeqrf_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnSgeqrf_bufferSize-NEXT:                               n /*int*/, a /*float **/, lda /*int*/,
// cusolverDnSgeqrf_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnSgeqrf_bufferSize-NEXT: Is migrated to:
// cusolverDnSgeqrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnSgeqrf_bufferSize-NEXT:   buffer_size = oneapi::mkl::lapack::geqrf_scratchpad_size<float>(*handle, m, n, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDgeqrf_bufferSize | FileCheck %s -check-prefix=cusolverDnDgeqrf_bufferSize
// cusolverDnDgeqrf_bufferSize: CUDA API:
// cusolverDnDgeqrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnDgeqrf_bufferSize-NEXT:   cusolverDnDgeqrf_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnDgeqrf_bufferSize-NEXT:                               n /*int*/, a /*double **/, lda /*int*/,
// cusolverDnDgeqrf_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnDgeqrf_bufferSize-NEXT: Is migrated to:
// cusolverDnDgeqrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnDgeqrf_bufferSize-NEXT:   buffer_size = oneapi::mkl::lapack::geqrf_scratchpad_size<double>(*handle, m, n, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCgeqrf_bufferSize | FileCheck %s -check-prefix=cusolverDnCgeqrf_bufferSize
// cusolverDnCgeqrf_bufferSize: CUDA API:
// cusolverDnCgeqrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnCgeqrf_bufferSize-NEXT:   cusolverDnCgeqrf_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnCgeqrf_bufferSize-NEXT:                               n /*int*/, a /*cuComplex **/, lda /*int*/,
// cusolverDnCgeqrf_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnCgeqrf_bufferSize-NEXT: Is migrated to:
// cusolverDnCgeqrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnCgeqrf_bufferSize-NEXT:   buffer_size = oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<float>>(*handle, m, n, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZgeqrf_bufferSize | FileCheck %s -check-prefix=cusolverDnZgeqrf_bufferSize
// cusolverDnZgeqrf_bufferSize: CUDA API:
// cusolverDnZgeqrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnZgeqrf_bufferSize-NEXT:   cusolverDnZgeqrf_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnZgeqrf_bufferSize-NEXT:                               n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZgeqrf_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnZgeqrf_bufferSize-NEXT: Is migrated to:
// cusolverDnZgeqrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnZgeqrf_bufferSize-NEXT:   buffer_size = oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<double>>(*handle, m, n, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSgeqrf | FileCheck %s -check-prefix=cusolverDnSgeqrf
// cusolverDnSgeqrf: CUDA API:
// cusolverDnSgeqrf-NEXT:   cusolverDnSgeqrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnSgeqrf-NEXT:                    a /*float **/, lda /*int*/, tau /*float **/,
// cusolverDnSgeqrf-NEXT:                    buffer /*float **/, buffer_size /*int*/, info /*int **/);
// cusolverDnSgeqrf-NEXT: Is migrated to:
// cusolverDnSgeqrf-NEXT:   oneapi::mkl::lapack::geqrf(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnSgeqrf-NEXT:                    (float*)a /*float **/, lda /*int*/, (float*)tau /*float **/,
// cusolverDnSgeqrf-NEXT:                    (float*)buffer /*float **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDgeqrf | FileCheck %s -check-prefix=cusolverDnDgeqrf
// cusolverDnDgeqrf: CUDA API:
// cusolverDnDgeqrf-NEXT:   cusolverDnDgeqrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnDgeqrf-NEXT:                    a /*double **/, lda /*int*/, tau /*double **/,
// cusolverDnDgeqrf-NEXT:                    buffer /*double **/, buffer_size /*int*/, info /*int **/);
// cusolverDnDgeqrf-NEXT: Is migrated to:
// cusolverDnDgeqrf-NEXT:   oneapi::mkl::lapack::geqrf(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnDgeqrf-NEXT:                    (double*)a /*double **/, lda /*int*/, (double*)tau /*double **/,
// cusolverDnDgeqrf-NEXT:                    (double*)buffer /*double **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCgeqrf | FileCheck %s -check-prefix=cusolverDnCgeqrf
// cusolverDnCgeqrf: CUDA API:
// cusolverDnCgeqrf-NEXT:   cusolverDnCgeqrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnCgeqrf-NEXT:                    a /*cuComplex **/, lda /*int*/, tau /*cuComplex **/,
// cusolverDnCgeqrf-NEXT:                    buffer /*cuComplex **/, buffer_size /*int*/, info /*int **/);
// cusolverDnCgeqrf-NEXT: Is migrated to:
// cusolverDnCgeqrf-NEXT:   oneapi::mkl::lapack::geqrf(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnCgeqrf-NEXT:                    (std::complex<float>*)a /*cuComplex **/, lda /*int*/, (std::complex<float>*)tau /*cuComplex **/,
// cusolverDnCgeqrf-NEXT:                    (std::complex<float>*)buffer /*cuComplex **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZgeqrf | FileCheck %s -check-prefix=cusolverDnZgeqrf
// cusolverDnZgeqrf: CUDA API:
// cusolverDnZgeqrf-NEXT:   cusolverDnZgeqrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnZgeqrf-NEXT:                    a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZgeqrf-NEXT:                    tau /*cuDoubleComplex **/, buffer /*cuDoubleComplex **/,
// cusolverDnZgeqrf-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnZgeqrf-NEXT: Is migrated to:
// cusolverDnZgeqrf-NEXT:   oneapi::mkl::lapack::geqrf(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnZgeqrf-NEXT:                    (std::complex<double>*)a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZgeqrf-NEXT:                    (std::complex<double>*)tau /*cuDoubleComplex **/, (std::complex<double>*)buffer /*cuDoubleComplex **/,
// cusolverDnZgeqrf-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSormqr_bufferSize | FileCheck %s -check-prefix=cusolverDnSormqr_bufferSize
// cusolverDnSormqr_bufferSize: CUDA API:
// cusolverDnSormqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnSormqr_bufferSize-NEXT:   cusolverDnSormqr_bufferSize(
// cusolverDnSormqr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnSormqr_bufferSize-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusolverDnSormqr_bufferSize-NEXT:       a /*const float **/, lda /*int*/, tau /*const float **/,
// cusolverDnSormqr_bufferSize-NEXT:       c /*const float **/, ldc /*int*/, &buffer_size /*int **/);
// cusolverDnSormqr_bufferSize-NEXT: Is migrated to:
// cusolverDnSormqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnSormqr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::ormqr_scratchpad_size<float>(
// cusolverDnSormqr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnSormqr_bufferSize-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*const float **/, lda /*const float **/, ldc /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDormqr_bufferSize | FileCheck %s -check-prefix=cusolverDnDormqr_bufferSize
// cusolverDnDormqr_bufferSize: CUDA API:
// cusolverDnDormqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnDormqr_bufferSize-NEXT:   cusolverDnDormqr_bufferSize(
// cusolverDnDormqr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnDormqr_bufferSize-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusolverDnDormqr_bufferSize-NEXT:       a /*const double **/, lda /*int*/, tau /*const double **/,
// cusolverDnDormqr_bufferSize-NEXT:       c /*const double **/, ldc /*int*/, &buffer_size /*int **/);
// cusolverDnDormqr_bufferSize-NEXT: Is migrated to:
// cusolverDnDormqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnDormqr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::ormqr_scratchpad_size<double>(
// cusolverDnDormqr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnDormqr_bufferSize-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*const double **/, lda /*const double **/, ldc /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCunmqr_bufferSize | FileCheck %s -check-prefix=cusolverDnCunmqr_bufferSize
// cusolverDnCunmqr_bufferSize: CUDA API:
// cusolverDnCunmqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnCunmqr_bufferSize-NEXT:   cusolverDnCunmqr_bufferSize(
// cusolverDnCunmqr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnCunmqr_bufferSize-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusolverDnCunmqr_bufferSize-NEXT:       a /*const cuComplex **/, lda /*int*/, tau /*const cuComplex **/,
// cusolverDnCunmqr_bufferSize-NEXT:       c /*const cuComplex **/, ldc /*int*/, &buffer_size /*int **/);
// cusolverDnCunmqr_bufferSize-NEXT: Is migrated to:
// cusolverDnCunmqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnCunmqr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<float>>(
// cusolverDnCunmqr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnCunmqr_bufferSize-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*const cuComplex **/, lda /*const cuComplex **/, ldc /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZunmqr_bufferSize | FileCheck %s -check-prefix=cusolverDnZunmqr_bufferSize
// cusolverDnZunmqr_bufferSize: CUDA API:
// cusolverDnZunmqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnZunmqr_bufferSize-NEXT:   cusolverDnZunmqr_bufferSize(
// cusolverDnZunmqr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnZunmqr_bufferSize-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusolverDnZunmqr_bufferSize-NEXT:       a /*const cuDoubleComplex **/, lda /*int*/,
// cusolverDnZunmqr_bufferSize-NEXT:       tau /*const cuDoubleComplex **/, c /*const cuDoubleComplex **/,
// cusolverDnZunmqr_bufferSize-NEXT:       ldc /*int*/, &buffer_size /*int **/);
// cusolverDnZunmqr_bufferSize-NEXT: Is migrated to:
// cusolverDnZunmqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnZunmqr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<double>>(
// cusolverDnZunmqr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnZunmqr_bufferSize-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*const cuDoubleComplex **/, lda /*const cuDoubleComplex **/,
// cusolverDnZunmqr_bufferSize-NEXT:       ldc /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSormqr | FileCheck %s -check-prefix=cusolverDnSormqr
// cusolverDnSormqr: CUDA API:
// cusolverDnSormqr-NEXT:   cusolverDnSormqr(
// cusolverDnSormqr-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnSormqr-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusolverDnSormqr-NEXT:       a /*const float **/, lda /*int*/, tau /*const float **/, c /*float **/,
// cusolverDnSormqr-NEXT:       ldc /*int*/, buffer /*float **/, buffer_size /*int*/, info /*int **/);
// cusolverDnSormqr-NEXT: Is migrated to:
// cusolverDnSormqr-NEXT:   oneapi::mkl::lapack::ormqr(
// cusolverDnSormqr-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnSormqr-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusolverDnSormqr-NEXT:       (float*)a /*const float **/, lda /*int*/, (float*)tau /*const float **/, (float*)c /*float **/,
// cusolverDnSormqr-NEXT:       ldc /*int*/, (float*)buffer /*float **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDormqr | FileCheck %s -check-prefix=cusolverDnDormqr
// cusolverDnDormqr: CUDA API:
// cusolverDnDormqr-NEXT:   cusolverDnDormqr(
// cusolverDnDormqr-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnDormqr-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusolverDnDormqr-NEXT:       a /*const double **/, lda /*int*/, tau /*const double **/, c /*double **/,
// cusolverDnDormqr-NEXT:       ldc /*int*/, buffer /*double **/, buffer_size /*int*/, info /*int **/);
// cusolverDnDormqr-NEXT: Is migrated to:
// cusolverDnDormqr-NEXT:   oneapi::mkl::lapack::ormqr(
// cusolverDnDormqr-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnDormqr-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusolverDnDormqr-NEXT:       (double*)a /*const double **/, lda /*int*/, (double*)tau /*const double **/, (double*)c /*double **/,
// cusolverDnDormqr-NEXT:       ldc /*int*/, (double*)buffer /*double **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCunmqr | FileCheck %s -check-prefix=cusolverDnCunmqr
// cusolverDnCunmqr: CUDA API:
// cusolverDnCunmqr-NEXT:   cusolverDnCunmqr(handle /*cusolverDnHandle_t*/,
// cusolverDnCunmqr-NEXT:                    left_right /*cublasSideMode_t*/, trans /*cublasOperation_t*/,
// cusolverDnCunmqr-NEXT:                    m /*int*/, n /*int*/, k /*int*/, a /*const cuComplex **/,
// cusolverDnCunmqr-NEXT:                    lda /*int*/, tau /*const cuComplex **/, c /*cuComplex **/,
// cusolverDnCunmqr-NEXT:                    ldc /*int*/, buffer /*cuComplex **/, buffer_size /*int*/,
// cusolverDnCunmqr-NEXT:                    info /*int **/);
// cusolverDnCunmqr-NEXT: Is migrated to:
// cusolverDnCunmqr-NEXT:   oneapi::mkl::lapack::unmqr(*handle /*cusolverDnHandle_t*/,
// cusolverDnCunmqr-NEXT:                    left_right /*cublasSideMode_t*/, trans /*cublasOperation_t*/,
// cusolverDnCunmqr-NEXT:                    m /*int*/, n /*int*/, k /*int*/, (std::complex<float>*)a /*const cuComplex **/,
// cusolverDnCunmqr-NEXT:                    lda /*int*/, (std::complex<float>*)tau /*const cuComplex **/, (std::complex<float>*)c /*cuComplex **/,
// cusolverDnCunmqr-NEXT:                    ldc /*int*/, (std::complex<float>*)buffer /*cuComplex **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZunmqr | FileCheck %s -check-prefix=cusolverDnZunmqr
// cusolverDnZunmqr: CUDA API:
// cusolverDnZunmqr-NEXT:   cusolverDnZunmqr(
// cusolverDnZunmqr-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnZunmqr-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusolverDnZunmqr-NEXT:       a /*const cuDoubleComplex **/, lda /*int*/,
// cusolverDnZunmqr-NEXT:       tau /*const cuDoubleComplex **/, c /*cuDoubleComplex **/, ldc /*int*/,
// cusolverDnZunmqr-NEXT:       buffer /*cuDoubleComplex **/, buffer_size /*int*/, info /*int **/);
// cusolverDnZunmqr-NEXT: Is migrated to:
// cusolverDnZunmqr-NEXT:   oneapi::mkl::lapack::unmqr(
// cusolverDnZunmqr-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnZunmqr-NEXT:       trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusolverDnZunmqr-NEXT:       (std::complex<double>*)a /*const cuDoubleComplex **/, lda /*int*/,
// cusolverDnZunmqr-NEXT:       (std::complex<double>*)tau /*const cuDoubleComplex **/, (std::complex<double>*)c /*cuDoubleComplex **/, ldc /*int*/,
// cusolverDnZunmqr-NEXT:       (std::complex<double>*)buffer /*cuDoubleComplex **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSorgqr_bufferSize | FileCheck %s -check-prefix=cusolverDnSorgqr_bufferSize
// cusolverDnSorgqr_bufferSize: CUDA API:
// cusolverDnSorgqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnSorgqr_bufferSize-NEXT:   cusolverDnSorgqr_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnSorgqr_bufferSize-NEXT:                               n /*int*/, k /*int*/, a /*const float **/,
// cusolverDnSorgqr_bufferSize-NEXT:                               lda /*int*/, tau /*const float **/,
// cusolverDnSorgqr_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnSorgqr_bufferSize-NEXT: Is migrated to:
// cusolverDnSorgqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnSorgqr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::orgqr_scratchpad_size<float>(*handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnSorgqr_bufferSize-NEXT:                               n /*int*/, k /*const float **/,
// cusolverDnSorgqr_bufferSize-NEXT:                               lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDorgqr_bufferSize | FileCheck %s -check-prefix=cusolverDnDorgqr_bufferSize
// cusolverDnDorgqr_bufferSize: CUDA API:
// cusolverDnDorgqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnDorgqr_bufferSize-NEXT:   cusolverDnDorgqr_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnDorgqr_bufferSize-NEXT:                               n /*int*/, k /*int*/, a /*const double **/,
// cusolverDnDorgqr_bufferSize-NEXT:                               lda /*int*/, tau /*const double **/,
// cusolverDnDorgqr_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnDorgqr_bufferSize-NEXT: Is migrated to:
// cusolverDnDorgqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnDorgqr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::orgqr_scratchpad_size<double>(*handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnDorgqr_bufferSize-NEXT:                               n /*int*/, k /*const double **/,
// cusolverDnDorgqr_bufferSize-NEXT:                               lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCungqr_bufferSize | FileCheck %s -check-prefix=cusolverDnCungqr_bufferSize
// cusolverDnCungqr_bufferSize: CUDA API:
// cusolverDnCungqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnCungqr_bufferSize-NEXT:   cusolverDnCungqr_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnCungqr_bufferSize-NEXT:                               n /*int*/, k /*int*/, a /*const cuComplex **/,
// cusolverDnCungqr_bufferSize-NEXT:                               lda /*int*/, tau /*const cuComplex **/,
// cusolverDnCungqr_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnCungqr_bufferSize-NEXT: Is migrated to:
// cusolverDnCungqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnCungqr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::ungqr_scratchpad_size<std::complex<float>>(*handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnCungqr_bufferSize-NEXT:                               n /*int*/, k /*const cuComplex **/,
// cusolverDnCungqr_bufferSize-NEXT:                               lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZungqr_bufferSize | FileCheck %s -check-prefix=cusolverDnZungqr_bufferSize
// cusolverDnZungqr_bufferSize: CUDA API:
// cusolverDnZungqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnZungqr_bufferSize-NEXT:   cusolverDnZungqr_bufferSize(
// cusolverDnZungqr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusolverDnZungqr_bufferSize-NEXT:       a /*const cuDoubleComplex **/, lda /*int*/,
// cusolverDnZungqr_bufferSize-NEXT:       tau /*const cuDoubleComplex **/, &buffer_size /*int **/);
// cusolverDnZungqr_bufferSize-NEXT: Is migrated to:
// cusolverDnZungqr_bufferSize-NEXT:   int buffer_size;
// cusolverDnZungqr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::ungqr_scratchpad_size<std::complex<double>>(
// cusolverDnZungqr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/, k /*const cuDoubleComplex **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSorgqr | FileCheck %s -check-prefix=cusolverDnSorgqr
// cusolverDnSorgqr: CUDA API:
// cusolverDnSorgqr-NEXT:   cusolverDnSorgqr(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnSorgqr-NEXT:                    k /*int*/, a /*float **/, lda /*int*/, tau /*const float **/,
// cusolverDnSorgqr-NEXT:                    buffer /*float **/, buffer_size /*int*/, info /*int **/);
// cusolverDnSorgqr-NEXT: Is migrated to:
// cusolverDnSorgqr-NEXT:   oneapi::mkl::lapack::orgqr(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnSorgqr-NEXT:                    k /*int*/, (float*)a /*float **/, lda /*int*/, (float*)tau /*const float **/,
// cusolverDnSorgqr-NEXT:                    (float*)buffer /*float **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDorgqr | FileCheck %s -check-prefix=cusolverDnDorgqr
// cusolverDnDorgqr: CUDA API:
// cusolverDnDorgqr-NEXT:   cusolverDnDorgqr(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnDorgqr-NEXT:                    k /*int*/, a /*double **/, lda /*int*/,
// cusolverDnDorgqr-NEXT:                    tau /*const double **/, buffer /*double **/,
// cusolverDnDorgqr-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnDorgqr-NEXT: Is migrated to:
// cusolverDnDorgqr-NEXT:   oneapi::mkl::lapack::orgqr(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnDorgqr-NEXT:                    k /*int*/, (double*)a /*double **/, lda /*int*/,
// cusolverDnDorgqr-NEXT:                    (double*)tau /*const double **/, (double*)buffer /*double **/,
// cusolverDnDorgqr-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCungqr | FileCheck %s -check-prefix=cusolverDnCungqr
// cusolverDnCungqr: CUDA API:
// cusolverDnCungqr-NEXT:   cusolverDnCungqr(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnCungqr-NEXT:                    k /*int*/, a /*cuComplex **/, lda /*int*/,
// cusolverDnCungqr-NEXT:                    tau /*const cuComplex **/, buffer /*cuComplex **/,
// cusolverDnCungqr-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnCungqr-NEXT: Is migrated to:
// cusolverDnCungqr-NEXT:   oneapi::mkl::lapack::ungqr(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnCungqr-NEXT:                    k /*int*/, (std::complex<float>*)a /*cuComplex **/, lda /*int*/,
// cusolverDnCungqr-NEXT:                    (std::complex<float>*)tau /*const cuComplex **/, (std::complex<float>*)buffer /*cuComplex **/,
// cusolverDnCungqr-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZungqr | FileCheck %s -check-prefix=cusolverDnZungqr
// cusolverDnZungqr: CUDA API:
// cusolverDnZungqr-NEXT:   cusolverDnZungqr(
// cusolverDnZungqr-NEXT:       handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusolverDnZungqr-NEXT:       a /*cuDoubleComplex **/, lda /*int*/, tau /*const cuDoubleComplex **/,
// cusolverDnZungqr-NEXT:       buffer /*cuDoubleComplex **/, buffer_size /*int*/, info /*int **/);
// cusolverDnZungqr-NEXT: Is migrated to:
// cusolverDnZungqr-NEXT:   oneapi::mkl::lapack::ungqr(
// cusolverDnZungqr-NEXT:       *handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusolverDnZungqr-NEXT:       (std::complex<double>*)a /*cuDoubleComplex **/, lda /*int*/, (std::complex<double>*)tau /*const cuDoubleComplex **/,
// cusolverDnZungqr-NEXT:       (std::complex<double>*)buffer /*cuDoubleComplex **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsytrf_bufferSize | FileCheck %s -check-prefix=cusolverDnSsytrf_bufferSize
// cusolverDnSsytrf_bufferSize: CUDA API:
// cusolverDnSsytrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsytrf_bufferSize-NEXT:   cusolverDnSsytrf_bufferSize(handle /*cusolverDnHandle_t*/, n /*int*/,
// cusolverDnSsytrf_bufferSize-NEXT:                               a /*float **/, lda /*int*/,
// cusolverDnSsytrf_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnSsytrf_bufferSize-NEXT: Is migrated to:
// cusolverDnSsytrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsytrf_bufferSize-NEXT:   {
// cusolverDnSsytrf_bufferSize-NEXT:   oneapi::mkl::uplo uplo_ct_mkl_upper_lower;
// cusolverDnSsytrf_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::sytrf_scratchpad_size<float>(*handle, uplo_ct_mkl_upper_lower /*cusolverDnHandle_t*/, n /*float **/, lda /*int **/);
// cusolverDnSsytrf_bufferSize-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsytrf_bufferSize | FileCheck %s -check-prefix=cusolverDnDsytrf_bufferSize
// cusolverDnDsytrf_bufferSize: CUDA API:
// cusolverDnDsytrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsytrf_bufferSize-NEXT:   cusolverDnDsytrf_bufferSize(handle /*cusolverDnHandle_t*/, n /*int*/,
// cusolverDnDsytrf_bufferSize-NEXT:                               a /*double **/, lda /*int*/,
// cusolverDnDsytrf_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnDsytrf_bufferSize-NEXT: Is migrated to:
// cusolverDnDsytrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsytrf_bufferSize-NEXT:   {
// cusolverDnDsytrf_bufferSize-NEXT:   oneapi::mkl::uplo uplo_ct_mkl_upper_lower;
// cusolverDnDsytrf_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::sytrf_scratchpad_size<double>(*handle, uplo_ct_mkl_upper_lower /*cusolverDnHandle_t*/, n /*double **/, lda /*int **/);
// cusolverDnDsytrf_bufferSize-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCsytrf_bufferSize | FileCheck %s -check-prefix=cusolverDnCsytrf_bufferSize
// cusolverDnCsytrf_bufferSize: CUDA API:
// cusolverDnCsytrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnCsytrf_bufferSize-NEXT:   cusolverDnCsytrf_bufferSize(handle /*cusolverDnHandle_t*/, n /*int*/,
// cusolverDnCsytrf_bufferSize-NEXT:                               a /*cuComplex **/, lda /*int*/,
// cusolverDnCsytrf_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnCsytrf_bufferSize-NEXT: Is migrated to:
// cusolverDnCsytrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnCsytrf_bufferSize-NEXT:   {
// cusolverDnCsytrf_bufferSize-NEXT:   oneapi::mkl::uplo uplo_ct_mkl_upper_lower;
// cusolverDnCsytrf_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<float>>(*handle, uplo_ct_mkl_upper_lower /*cusolverDnHandle_t*/, n /*cuComplex **/, lda /*int **/);
// cusolverDnCsytrf_bufferSize-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZsytrf_bufferSize | FileCheck %s -check-prefix=cusolverDnZsytrf_bufferSize
// cusolverDnZsytrf_bufferSize: CUDA API:
// cusolverDnZsytrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnZsytrf_bufferSize-NEXT:   cusolverDnZsytrf_bufferSize(handle /*cusolverDnHandle_t*/, n /*int*/,
// cusolverDnZsytrf_bufferSize-NEXT:                               a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZsytrf_bufferSize-NEXT:                               &buffer_size /*int **/);
// cusolverDnZsytrf_bufferSize-NEXT: Is migrated to:
// cusolverDnZsytrf_bufferSize-NEXT:   int buffer_size;
// cusolverDnZsytrf_bufferSize-NEXT:   {
// cusolverDnZsytrf_bufferSize-NEXT:   oneapi::mkl::uplo uplo_ct_mkl_upper_lower;
// cusolverDnZsytrf_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<double>>(*handle, uplo_ct_mkl_upper_lower /*cusolverDnHandle_t*/, n /*cuDoubleComplex **/, lda /*int **/);
// cusolverDnZsytrf_bufferSize-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsytrf | FileCheck %s -check-prefix=cusolverDnSsytrf
// cusolverDnSsytrf: CUDA API:
// cusolverDnSsytrf-NEXT:   cusolverDnSsytrf(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSsytrf-NEXT:                    n /*int*/, a /*float **/, lda /*int*/, ipiv /*int **/,
// cusolverDnSsytrf-NEXT:                    buffer /*float **/, buffer_size /*int*/, info /*int **/);
// cusolverDnSsytrf-NEXT: Is migrated to:
// cusolverDnSsytrf-NEXT:   {
// cusolverDnSsytrf-NEXT:   int64_t result_temp_pointer5;
// cusolverDnSsytrf-NEXT:   oneapi::mkl::lapack::sytrf(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSsytrf-NEXT:                    n /*int*/, (float*)a /*float **/, lda /*int*/, &result_temp_pointer5 /*int **/,
// cusolverDnSsytrf-NEXT:                    (float*)buffer /*float **/, buffer_size /*int **/);
// cusolverDnSsytrf-NEXT:    *ipiv = result_temp_pointer5;
// cusolverDnSsytrf-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsytrf | FileCheck %s -check-prefix=cusolverDnDsytrf
// cusolverDnDsytrf: CUDA API:
// cusolverDnDsytrf-NEXT:   cusolverDnDsytrf(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDsytrf-NEXT:                    n /*int*/, a /*double **/, lda /*int*/, ipiv /*int **/,
// cusolverDnDsytrf-NEXT:                    buffer /*double **/, buffer_size /*int*/, info /*int **/);
// cusolverDnDsytrf-NEXT: Is migrated to:
// cusolverDnDsytrf-NEXT:   {
// cusolverDnDsytrf-NEXT:   int64_t result_temp_pointer5;
// cusolverDnDsytrf-NEXT:   oneapi::mkl::lapack::sytrf(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDsytrf-NEXT:                    n /*int*/, (double*)a /*double **/, lda /*int*/, &result_temp_pointer5 /*int **/,
// cusolverDnDsytrf-NEXT:                    (double*)buffer /*double **/, buffer_size /*int **/);
// cusolverDnDsytrf-NEXT:    *ipiv = result_temp_pointer5;
// cusolverDnDsytrf-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCsytrf | FileCheck %s -check-prefix=cusolverDnCsytrf
// cusolverDnCsytrf: CUDA API:
// cusolverDnCsytrf-NEXT:   cusolverDnCsytrf(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnCsytrf-NEXT:                    n /*int*/, a /*cuComplex **/, lda /*int*/, ipiv /*int **/,
// cusolverDnCsytrf-NEXT:                    buffer /*cuComplex **/, buffer_size /*int*/, info /*int **/);
// cusolverDnCsytrf-NEXT: Is migrated to:
// cusolverDnCsytrf-NEXT:   {
// cusolverDnCsytrf-NEXT:   int64_t result_temp_pointer5;
// cusolverDnCsytrf-NEXT:   oneapi::mkl::lapack::sytrf(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnCsytrf-NEXT:                    n /*int*/, (std::complex<float>*)a /*cuComplex **/, lda /*int*/, &result_temp_pointer5 /*int **/,
// cusolverDnCsytrf-NEXT:                    (std::complex<float>*)buffer /*cuComplex **/, buffer_size /*int **/);
// cusolverDnCsytrf-NEXT:    *ipiv = result_temp_pointer5;
// cusolverDnCsytrf-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZsytrf | FileCheck %s -check-prefix=cusolverDnZsytrf
// cusolverDnZsytrf: CUDA API:
// cusolverDnZsytrf-NEXT:   cusolverDnZsytrf(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnZsytrf-NEXT:                    n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZsytrf-NEXT:                    ipiv /*int **/, buffer /*cuDoubleComplex **/,
// cusolverDnZsytrf-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnZsytrf-NEXT: Is migrated to:
// cusolverDnZsytrf-NEXT:   {
// cusolverDnZsytrf-NEXT:   int64_t result_temp_pointer5;
// cusolverDnZsytrf-NEXT:   oneapi::mkl::lapack::sytrf(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnZsytrf-NEXT:                    n /*int*/, (std::complex<double>*)a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZsytrf-NEXT:                    &result_temp_pointer5 /*int **/, (std::complex<double>*)buffer /*cuDoubleComplex **/,
// cusolverDnZsytrf-NEXT:                    buffer_size /*int **/);
// cusolverDnZsytrf-NEXT:    *ipiv = result_temp_pointer5;
// cusolverDnZsytrf-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSgebrd_bufferSize | FileCheck %s -check-prefix=cusolverDnSgebrd_bufferSize
// cusolverDnSgebrd_bufferSize: CUDA API:
// cusolverDnSgebrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnSgebrd_bufferSize-NEXT:   cusolverDnSgebrd_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnSgebrd_bufferSize-NEXT:                               n /*int*/, &buffer_size /*int **/);
// cusolverDnSgebrd_bufferSize-NEXT: Is migrated to:
// cusolverDnSgebrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnSgebrd_bufferSize-NEXT:   {
// cusolverDnSgebrd_bufferSize-NEXT:   std::int64_t lda_ct;
// cusolverDnSgebrd_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::gebrd_scratchpad_size<float>(*handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnSgebrd_bufferSize-NEXT:                               n, lda_ct /*int **/);
// cusolverDnSgebrd_bufferSize-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDgebrd_bufferSize | FileCheck %s -check-prefix=cusolverDnDgebrd_bufferSize
// cusolverDnDgebrd_bufferSize: CUDA API:
// cusolverDnDgebrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnDgebrd_bufferSize-NEXT:   cusolverDnDgebrd_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnDgebrd_bufferSize-NEXT:                               n /*int*/, &buffer_size /*int **/);
// cusolverDnDgebrd_bufferSize-NEXT: Is migrated to:
// cusolverDnDgebrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnDgebrd_bufferSize-NEXT:   {
// cusolverDnDgebrd_bufferSize-NEXT:   std::int64_t lda_ct;
// cusolverDnDgebrd_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::gebrd_scratchpad_size<double>(*handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnDgebrd_bufferSize-NEXT:                               n, lda_ct /*int **/);
// cusolverDnDgebrd_bufferSize-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCgebrd_bufferSize | FileCheck %s -check-prefix=cusolverDnCgebrd_bufferSize
// cusolverDnCgebrd_bufferSize: CUDA API:
// cusolverDnCgebrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnCgebrd_bufferSize-NEXT:   cusolverDnCgebrd_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnCgebrd_bufferSize-NEXT:                               n /*int*/, &buffer_size /*int **/);
// cusolverDnCgebrd_bufferSize-NEXT: Is migrated to:
// cusolverDnCgebrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnCgebrd_bufferSize-NEXT:   {
// cusolverDnCgebrd_bufferSize-NEXT:   std::int64_t lda_ct;
// cusolverDnCgebrd_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<float>>(*handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnCgebrd_bufferSize-NEXT:                               n, lda_ct /*int **/);
// cusolverDnCgebrd_bufferSize-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZgebrd_bufferSize | FileCheck %s -check-prefix=cusolverDnZgebrd_bufferSize
// cusolverDnZgebrd_bufferSize: CUDA API:
// cusolverDnZgebrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnZgebrd_bufferSize-NEXT:   cusolverDnZgebrd_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnZgebrd_bufferSize-NEXT:                               n /*int*/, &buffer_size /*int **/);
// cusolverDnZgebrd_bufferSize-NEXT: Is migrated to:
// cusolverDnZgebrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnZgebrd_bufferSize-NEXT:   {
// cusolverDnZgebrd_bufferSize-NEXT:   std::int64_t lda_ct;
// cusolverDnZgebrd_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<double>>(*handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnZgebrd_bufferSize-NEXT:                               n, lda_ct /*int **/);
// cusolverDnZgebrd_bufferSize-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSgebrd | FileCheck %s -check-prefix=cusolverDnSgebrd
// cusolverDnSgebrd: CUDA API:
// cusolverDnSgebrd-NEXT:   cusolverDnSgebrd(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnSgebrd-NEXT:                    a /*float **/, lda /*int*/, d /*float **/, e /*float **/,
// cusolverDnSgebrd-NEXT:                    tau_q /*float **/, tau_p /*float **/, buffer /*float **/,
// cusolverDnSgebrd-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnSgebrd-NEXT: Is migrated to:
// cusolverDnSgebrd-NEXT:   oneapi::mkl::lapack::gebrd(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnSgebrd-NEXT:                    (float*)a /*float **/, lda /*int*/, (float*)d /*float **/, (float*)e /*float **/,
// cusolverDnSgebrd-NEXT:                    (float*)tau_q /*float **/, (float*)tau_p /*float **/, (float*)buffer /*float **/,
// cusolverDnSgebrd-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDgebrd | FileCheck %s -check-prefix=cusolverDnDgebrd
// cusolverDnDgebrd: CUDA API:
// cusolverDnDgebrd-NEXT:   cusolverDnDgebrd(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnDgebrd-NEXT:                    a /*double **/, lda /*int*/, d /*double **/, e /*double **/,
// cusolverDnDgebrd-NEXT:                    tau_q /*double **/, tau_p /*double **/, buffer /*double **/,
// cusolverDnDgebrd-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnDgebrd-NEXT: Is migrated to:
// cusolverDnDgebrd-NEXT:   oneapi::mkl::lapack::gebrd(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnDgebrd-NEXT:                    (double*)a /*double **/, lda /*int*/, (double*)d /*double **/, (double*)e /*double **/,
// cusolverDnDgebrd-NEXT:                    (double*)tau_q /*double **/, (double*)tau_p /*double **/, (double*)buffer /*double **/,
// cusolverDnDgebrd-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCgebrd | FileCheck %s -check-prefix=cusolverDnCgebrd
// cusolverDnCgebrd: CUDA API:
// cusolverDnCgebrd-NEXT:   cusolverDnCgebrd(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnCgebrd-NEXT:                    a /*cuComplex **/, lda /*int*/, d /*float **/, e /*float **/,
// cusolverDnCgebrd-NEXT:                    tau_q /*cuComplex **/, tau_p /*cuComplex **/,
// cusolverDnCgebrd-NEXT:                    buffer /*cuComplex **/, buffer_size /*int*/, info /*int **/);
// cusolverDnCgebrd-NEXT: Is migrated to:
// cusolverDnCgebrd-NEXT:   oneapi::mkl::lapack::gebrd(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnCgebrd-NEXT:                    (std::complex<float>*)a /*cuComplex **/, lda /*int*/, (float*)d /*float **/, (float*)e /*float **/,
// cusolverDnCgebrd-NEXT:                    (std::complex<float>*)tau_q /*cuComplex **/, (std::complex<float>*)tau_p /*cuComplex **/,
// cusolverDnCgebrd-NEXT:                    (std::complex<float>*)buffer /*cuComplex **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZgebrd | FileCheck %s -check-prefix=cusolverDnZgebrd
// cusolverDnZgebrd: CUDA API:
// cusolverDnZgebrd-NEXT:   cusolverDnZgebrd(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnZgebrd-NEXT:                    a /*cuDoubleComplex **/, lda /*int*/, d /*double **/,
// cusolverDnZgebrd-NEXT:                    e /*double **/, tau_q /*cuDoubleComplex **/,
// cusolverDnZgebrd-NEXT:                    tau_p /*cuDoubleComplex **/, buffer /*cuDoubleComplex **/,
// cusolverDnZgebrd-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnZgebrd-NEXT: Is migrated to:
// cusolverDnZgebrd-NEXT:   oneapi::mkl::lapack::gebrd(*handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
// cusolverDnZgebrd-NEXT:                    (std::complex<double>*)a /*cuDoubleComplex **/, lda /*int*/, (double*)d /*double **/,
// cusolverDnZgebrd-NEXT:                    (double*)e /*double **/, (std::complex<double>*)tau_q /*cuDoubleComplex **/,
// cusolverDnZgebrd-NEXT:                    (std::complex<double>*)tau_p /*cuDoubleComplex **/, (std::complex<double>*)buffer /*cuDoubleComplex **/,
// cusolverDnZgebrd-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSorgbr_bufferSize | FileCheck %s -check-prefix=cusolverDnSorgbr_bufferSize
// cusolverDnSorgbr_bufferSize: CUDA API:
// cusolverDnSorgbr_bufferSize-NEXT:   int buffer_size;
// cusolverDnSorgbr_bufferSize-NEXT:   cusolverDnSorgbr_bufferSize(
// cusolverDnSorgbr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/, m /*int*/,
// cusolverDnSorgbr_bufferSize-NEXT:       n /*int*/, k /*int*/, a /*const float **/, lda /*int*/,
// cusolverDnSorgbr_bufferSize-NEXT:       tau /*const float **/, &buffer_size /*int **/);
// cusolverDnSorgbr_bufferSize-NEXT: Is migrated to:
// cusolverDnSorgbr_bufferSize-NEXT:   int buffer_size;
// cusolverDnSorgbr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::orgbr_scratchpad_size<float>(
// cusolverDnSorgbr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, (oneapi::mkl::generate)left_right /*cublasSideMode_t*/, m /*int*/,
// cusolverDnSorgbr_bufferSize-NEXT:       n /*int*/, k /*const float **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDorgbr_bufferSize | FileCheck %s -check-prefix=cusolverDnDorgbr_bufferSize
// cusolverDnDorgbr_bufferSize: CUDA API:
// cusolverDnDorgbr_bufferSize-NEXT:   int buffer_size;
// cusolverDnDorgbr_bufferSize-NEXT:   cusolverDnDorgbr_bufferSize(
// cusolverDnDorgbr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/, m /*int*/,
// cusolverDnDorgbr_bufferSize-NEXT:       n /*int*/, k /*int*/, a /*const double **/, lda /*int*/,
// cusolverDnDorgbr_bufferSize-NEXT:       tau /*const double **/, &buffer_size /*int **/);
// cusolverDnDorgbr_bufferSize-NEXT: Is migrated to:
// cusolverDnDorgbr_bufferSize-NEXT:   int buffer_size;
// cusolverDnDorgbr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::orgbr_scratchpad_size<double>(
// cusolverDnDorgbr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, (oneapi::mkl::generate)left_right /*cublasSideMode_t*/, m /*int*/,
// cusolverDnDorgbr_bufferSize-NEXT:       n /*int*/, k /*const double **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCungbr_bufferSize | FileCheck %s -check-prefix=cusolverDnCungbr_bufferSize
// cusolverDnCungbr_bufferSize: CUDA API:
// cusolverDnCungbr_bufferSize-NEXT:   int buffer_size;
// cusolverDnCungbr_bufferSize-NEXT:   cusolverDnCungbr_bufferSize(
// cusolverDnCungbr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/, m /*int*/,
// cusolverDnCungbr_bufferSize-NEXT:       n /*int*/, k /*int*/, a /*const cuComplex **/, lda /*int*/,
// cusolverDnCungbr_bufferSize-NEXT:       tau /*const cuComplex **/, &buffer_size /*int **/);
// cusolverDnCungbr_bufferSize-NEXT: Is migrated to:
// cusolverDnCungbr_bufferSize-NEXT:   int buffer_size;
// cusolverDnCungbr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::ungbr_scratchpad_size<std::complex<float>>(
// cusolverDnCungbr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, (oneapi::mkl::generate)left_right /*cublasSideMode_t*/, m /*int*/,
// cusolverDnCungbr_bufferSize-NEXT:       n /*int*/, k /*const cuComplex **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZungbr_bufferSize | FileCheck %s -check-prefix=cusolverDnZungbr_bufferSize
// cusolverDnZungbr_bufferSize: CUDA API:
// cusolverDnZungbr_bufferSize-NEXT:   int buffer_size;
// cusolverDnZungbr_bufferSize-NEXT:   cusolverDnZungbr_bufferSize(
// cusolverDnZungbr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/, m /*int*/,
// cusolverDnZungbr_bufferSize-NEXT:       n /*int*/, k /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
// cusolverDnZungbr_bufferSize-NEXT:       tau /*const cuDoubleComplex **/, &buffer_size /*int **/);
// cusolverDnZungbr_bufferSize-NEXT: Is migrated to:
// cusolverDnZungbr_bufferSize-NEXT:   int buffer_size;
// cusolverDnZungbr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::ungbr_scratchpad_size<std::complex<double>>(
// cusolverDnZungbr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, (oneapi::mkl::generate)left_right /*cublasSideMode_t*/, m /*int*/,
// cusolverDnZungbr_bufferSize-NEXT:       n /*int*/, k /*const cuDoubleComplex **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSorgbr | FileCheck %s -check-prefix=cusolverDnSorgbr
// cusolverDnSorgbr: CUDA API:
// cusolverDnSorgbr-NEXT:   cusolverDnSorgbr(handle /*cusolverDnHandle_t*/,
// cusolverDnSorgbr-NEXT:                    left_right /*cublasSideMode_t*/, m /*int*/, n /*int*/,
// cusolverDnSorgbr-NEXT:                    k /*int*/, a /*float **/, lda /*int*/, tau /*const float **/,
// cusolverDnSorgbr-NEXT:                    buffer /*float **/, buffer_size /*int*/, info /*int **/);
// cusolverDnSorgbr-NEXT: Is migrated to:
// cusolverDnSorgbr-NEXT:   oneapi::mkl::lapack::orgbr(*handle /*cusolverDnHandle_t*/,
// cusolverDnSorgbr-NEXT:                    (oneapi::mkl::generate)left_right /*cublasSideMode_t*/, m /*int*/, n /*int*/,
// cusolverDnSorgbr-NEXT:                    k /*int*/, (float*)a /*float **/, lda /*int*/, (float*)tau /*const float **/,
// cusolverDnSorgbr-NEXT:                    (float*)buffer /*float **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDorgbr | FileCheck %s -check-prefix=cusolverDnDorgbr
// cusolverDnDorgbr: CUDA API:
// cusolverDnDorgbr-NEXT:   cusolverDnDorgbr(
// cusolverDnDorgbr-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/, m /*int*/,
// cusolverDnDorgbr-NEXT:       n /*int*/, k /*int*/, a /*double **/, lda /*int*/, tau /*const double **/,
// cusolverDnDorgbr-NEXT:       buffer /*double **/, buffer_size /*int*/, info /*int **/);
// cusolverDnDorgbr-NEXT: Is migrated to:
// cusolverDnDorgbr-NEXT:   oneapi::mkl::lapack::orgbr(
// cusolverDnDorgbr-NEXT:       *handle /*cusolverDnHandle_t*/, (oneapi::mkl::generate)left_right /*cublasSideMode_t*/, m /*int*/,
// cusolverDnDorgbr-NEXT:       n /*int*/, k /*int*/, (double*)a /*double **/, lda /*int*/, (double*)tau /*const double **/,
// cusolverDnDorgbr-NEXT:       (double*)buffer /*double **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCungbr | FileCheck %s -check-prefix=cusolverDnCungbr
// cusolverDnCungbr: CUDA API:
// cusolverDnCungbr-NEXT:   cusolverDnCungbr(handle /*cusolverDnHandle_t*/,
// cusolverDnCungbr-NEXT:                    left_right /*cublasSideMode_t*/, m /*int*/, n /*int*/,
// cusolverDnCungbr-NEXT:                    k /*int*/, a /*cuComplex **/, lda /*int*/,
// cusolverDnCungbr-NEXT:                    tau /*const cuComplex **/, buffer /*cuComplex **/,
// cusolverDnCungbr-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnCungbr-NEXT: Is migrated to:
// cusolverDnCungbr-NEXT:   oneapi::mkl::lapack::ungbr(*handle /*cusolverDnHandle_t*/,
// cusolverDnCungbr-NEXT:                    (oneapi::mkl::generate)left_right /*cublasSideMode_t*/, m /*int*/, n /*int*/,
// cusolverDnCungbr-NEXT:                    k /*int*/, (std::complex<float>*)a /*cuComplex **/, lda /*int*/,
// cusolverDnCungbr-NEXT:                    (std::complex<float>*)tau /*const cuComplex **/, (std::complex<float>*)buffer /*cuComplex **/,
// cusolverDnCungbr-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZungbr | FileCheck %s -check-prefix=cusolverDnZungbr
// cusolverDnZungbr: CUDA API:
// cusolverDnZungbr-NEXT:   cusolverDnZungbr(
// cusolverDnZungbr-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/, m /*int*/,
// cusolverDnZungbr-NEXT:       n /*int*/, k /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZungbr-NEXT:       tau /*const cuDoubleComplex **/, buffer /*cuDoubleComplex **/,
// cusolverDnZungbr-NEXT:       buffer_size /*int*/, info /*int **/);
// cusolverDnZungbr-NEXT: Is migrated to:
// cusolverDnZungbr-NEXT:   oneapi::mkl::lapack::ungbr(
// cusolverDnZungbr-NEXT:       *handle /*cusolverDnHandle_t*/, (oneapi::mkl::generate)left_right /*cublasSideMode_t*/, m /*int*/,
// cusolverDnZungbr-NEXT:       n /*int*/, k /*int*/, (std::complex<double>*)a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZungbr-NEXT:       (std::complex<double>*)tau /*const cuDoubleComplex **/, (std::complex<double>*)buffer /*cuDoubleComplex **/,
// cusolverDnZungbr-NEXT:       buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsytrd_bufferSize | FileCheck %s -check-prefix=cusolverDnSsytrd_bufferSize
// cusolverDnSsytrd_bufferSize: CUDA API:
// cusolverDnSsytrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsytrd_bufferSize-NEXT:   cusolverDnSsytrd_bufferSize(
// cusolverDnSsytrd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnSsytrd_bufferSize-NEXT:       a /*const float **/, lda /*int*/, d /*const float **/,
// cusolverDnSsytrd_bufferSize-NEXT:       e /*const float **/, tau /*const float **/, &buffer_size /*int **/);
// cusolverDnSsytrd_bufferSize-NEXT: Is migrated to:
// cusolverDnSsytrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsytrd_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::sytrd_scratchpad_size<float>(
// cusolverDnSsytrd_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*const float **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsytrd_bufferSize | FileCheck %s -check-prefix=cusolverDnDsytrd_bufferSize
// cusolverDnDsytrd_bufferSize: CUDA API:
// cusolverDnDsytrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsytrd_bufferSize-NEXT:   cusolverDnDsytrd_bufferSize(
// cusolverDnDsytrd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnDsytrd_bufferSize-NEXT:       a /*const double **/, lda /*int*/, d /*const double **/,
// cusolverDnDsytrd_bufferSize-NEXT:       e /*const double **/, tau /*const double **/, &buffer_size /*int **/);
// cusolverDnDsytrd_bufferSize-NEXT: Is migrated to:
// cusolverDnDsytrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsytrd_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::sytrd_scratchpad_size<double>(
// cusolverDnDsytrd_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*const double **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnChetrd_bufferSize | FileCheck %s -check-prefix=cusolverDnChetrd_bufferSize
// cusolverDnChetrd_bufferSize: CUDA API:
// cusolverDnChetrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnChetrd_bufferSize-NEXT:   cusolverDnChetrd_bufferSize(
// cusolverDnChetrd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnChetrd_bufferSize-NEXT:       a /*const cuComplex **/, lda /*int*/, d /*const float **/,
// cusolverDnChetrd_bufferSize-NEXT:       e /*const float **/, tau /*const cuComplex **/, &buffer_size /*int **/);
// cusolverDnChetrd_bufferSize-NEXT: Is migrated to:
// cusolverDnChetrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnChetrd_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<float>>(
// cusolverDnChetrd_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*const cuComplex **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZhetrd_bufferSize | FileCheck %s -check-prefix=cusolverDnZhetrd_bufferSize
// cusolverDnZhetrd_bufferSize: CUDA API:
// cusolverDnZhetrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnZhetrd_bufferSize-NEXT:   cusolverDnZhetrd_bufferSize(
// cusolverDnZhetrd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZhetrd_bufferSize-NEXT:       a /*const cuDoubleComplex **/, lda /*int*/, d /*const double **/,
// cusolverDnZhetrd_bufferSize-NEXT:       e /*const double **/, tau /*const cuDoubleComplex **/,
// cusolverDnZhetrd_bufferSize-NEXT:       &buffer_size /*int **/);
// cusolverDnZhetrd_bufferSize-NEXT: Is migrated to:
// cusolverDnZhetrd_bufferSize-NEXT:   int buffer_size;
// cusolverDnZhetrd_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<double>>(
// cusolverDnZhetrd_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*const cuDoubleComplex **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsytrd | FileCheck %s -check-prefix=cusolverDnSsytrd
// cusolverDnSsytrd: CUDA API:
// cusolverDnSsytrd-NEXT:   cusolverDnSsytrd(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSsytrd-NEXT:                    n /*int*/, a /*float **/, lda /*int*/, d /*float **/,
// cusolverDnSsytrd-NEXT:                    e /*float **/, tau /*float **/, buffer /*float **/,
// cusolverDnSsytrd-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnSsytrd-NEXT: Is migrated to:
// cusolverDnSsytrd-NEXT:   oneapi::mkl::lapack::sytrd(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSsytrd-NEXT:                    n /*int*/, (float*)a /*float **/, lda /*int*/, (float*)d /*float **/,
// cusolverDnSsytrd-NEXT:                    (float*)e /*float **/, (float*)tau /*float **/, (float*)buffer /*float **/,
// cusolverDnSsytrd-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsytrd | FileCheck %s -check-prefix=cusolverDnDsytrd
// cusolverDnDsytrd: CUDA API:
// cusolverDnDsytrd-NEXT:   cusolverDnDsytrd(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDsytrd-NEXT:                    n /*int*/, a /*double **/, lda /*int*/, d /*double **/,
// cusolverDnDsytrd-NEXT:                    e /*double **/, tau /*double **/, buffer /*double **/,
// cusolverDnDsytrd-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnDsytrd-NEXT: Is migrated to:
// cusolverDnDsytrd-NEXT:   oneapi::mkl::lapack::sytrd(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDsytrd-NEXT:                    n /*int*/, (double*)a /*double **/, lda /*int*/, (double*)d /*double **/,
// cusolverDnDsytrd-NEXT:                    (double*)e /*double **/, (double*)tau /*double **/, (double*)buffer /*double **/,
// cusolverDnDsytrd-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnChetrd | FileCheck %s -check-prefix=cusolverDnChetrd
// cusolverDnChetrd: CUDA API:
// cusolverDnChetrd-NEXT:   cusolverDnChetrd(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnChetrd-NEXT:                    n /*int*/, a /*cuComplex **/, lda /*int*/, d /*float **/,
// cusolverDnChetrd-NEXT:                    e /*float **/, tau /*cuComplex **/, buffer /*cuComplex **/,
// cusolverDnChetrd-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnChetrd-NEXT: Is migrated to:
// cusolverDnChetrd-NEXT:   oneapi::mkl::lapack::hetrd(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnChetrd-NEXT:                    n /*int*/, (std::complex<float>*)a /*cuComplex **/, lda /*int*/, (float*)d /*float **/,
// cusolverDnChetrd-NEXT:                    (float*)e /*float **/, (std::complex<float>*)tau /*cuComplex **/, (std::complex<float>*)buffer /*cuComplex **/,
// cusolverDnChetrd-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZhetrd | FileCheck %s -check-prefix=cusolverDnZhetrd
// cusolverDnZhetrd: CUDA API:
// cusolverDnZhetrd-NEXT:   cusolverDnZhetrd(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnZhetrd-NEXT:                    n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZhetrd-NEXT:                    d /*double **/, e /*double **/, tau /*cuDoubleComplex **/,
// cusolverDnZhetrd-NEXT:                    buffer /*cuDoubleComplex **/, buffer_size /*int*/,
// cusolverDnZhetrd-NEXT:                    info /*int **/);
// cusolverDnZhetrd-NEXT: Is migrated to:
// cusolverDnZhetrd-NEXT:   oneapi::mkl::lapack::hetrd(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnZhetrd-NEXT:                    n /*int*/, (std::complex<double>*)a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZhetrd-NEXT:                    (double*)d /*double **/, (double*)e /*double **/, (std::complex<double>*)tau /*cuDoubleComplex **/,
// cusolverDnZhetrd-NEXT:                    (std::complex<double>*)buffer /*cuDoubleComplex **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSormtr_bufferSize | FileCheck %s -check-prefix=cusolverDnSormtr_bufferSize
// cusolverDnSormtr_bufferSize: CUDA API:
// cusolverDnSormtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnSormtr_bufferSize-NEXT:   cusolverDnSormtr_bufferSize(
// cusolverDnSormtr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnSormtr_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnSormtr_bufferSize-NEXT:       n /*int*/, a /*const float **/, lda /*int*/, tau /*const float **/,
// cusolverDnSormtr_bufferSize-NEXT:       c /*const float **/, ldc /*int*/, &buffer_size /*int **/);
// cusolverDnSormtr_bufferSize-NEXT: Is migrated to:
// cusolverDnSormtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnSormtr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::ormtr_scratchpad_size<float>(
// cusolverDnSormtr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnSormtr_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnSormtr_bufferSize-NEXT:       n /*const float **/, lda /*const float **/, ldc /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDormtr_bufferSize | FileCheck %s -check-prefix=cusolverDnDormtr_bufferSize
// cusolverDnDormtr_bufferSize: CUDA API:
// cusolverDnDormtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnDormtr_bufferSize-NEXT:   cusolverDnDormtr_bufferSize(
// cusolverDnDormtr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnDormtr_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnDormtr_bufferSize-NEXT:       n /*int*/, a /*const double **/, lda /*int*/, tau /*const double **/,
// cusolverDnDormtr_bufferSize-NEXT:       c /*const double **/, ldc /*int*/, &buffer_size /*int **/);
// cusolverDnDormtr_bufferSize-NEXT: Is migrated to:
// cusolverDnDormtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnDormtr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::ormtr_scratchpad_size<double>(
// cusolverDnDormtr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnDormtr_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnDormtr_bufferSize-NEXT:       n /*const double **/, lda /*const double **/, ldc /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCunmtr_bufferSize | FileCheck %s -check-prefix=cusolverDnCunmtr_bufferSize
// cusolverDnCunmtr_bufferSize: CUDA API:
// cusolverDnCunmtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnCunmtr_bufferSize-NEXT:   cusolverDnCunmtr_bufferSize(
// cusolverDnCunmtr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnCunmtr_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnCunmtr_bufferSize-NEXT:       n /*int*/, a /*const cuComplex **/, lda /*int*/,
// cusolverDnCunmtr_bufferSize-NEXT:       tau /*const cuComplex **/, c /*const cuComplex **/, ldc /*int*/,
// cusolverDnCunmtr_bufferSize-NEXT:       &buffer_size /*int **/);
// cusolverDnCunmtr_bufferSize-NEXT: Is migrated to:
// cusolverDnCunmtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnCunmtr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<float>>(
// cusolverDnCunmtr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnCunmtr_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnCunmtr_bufferSize-NEXT:       n /*const cuComplex **/, lda /*const cuComplex **/, ldc /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZunmtr_bufferSize | FileCheck %s -check-prefix=cusolverDnZunmtr_bufferSize
// cusolverDnZunmtr_bufferSize: CUDA API:
// cusolverDnZunmtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnZunmtr_bufferSize-NEXT:   cusolverDnZunmtr_bufferSize(
// cusolverDnZunmtr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnZunmtr_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnZunmtr_bufferSize-NEXT:       n /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
// cusolverDnZunmtr_bufferSize-NEXT:       tau /*const cuDoubleComplex **/, c /*const cuDoubleComplex **/,
// cusolverDnZunmtr_bufferSize-NEXT:       ldc /*int*/, &buffer_size /*int **/);
// cusolverDnZunmtr_bufferSize-NEXT: Is migrated to:
// cusolverDnZunmtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnZunmtr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<double>>(
// cusolverDnZunmtr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnZunmtr_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnZunmtr_bufferSize-NEXT:       n /*const cuDoubleComplex **/, lda /*const cuDoubleComplex **/,
// cusolverDnZunmtr_bufferSize-NEXT:       ldc /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSormtr | FileCheck %s -check-prefix=cusolverDnSormtr
// cusolverDnSormtr: CUDA API:
// cusolverDnSormtr-NEXT:   cusolverDnSormtr(
// cusolverDnSormtr-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnSormtr-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnSormtr-NEXT:       n /*int*/, a /*float **/, lda /*int*/, tau /*float **/, c /*float **/,
// cusolverDnSormtr-NEXT:       ldc /*int*/, buffer /*float **/, buffer_size /*int*/, info /*int **/);
// cusolverDnSormtr-NEXT: Is migrated to:
// cusolverDnSormtr-NEXT:   oneapi::mkl::lapack::ormtr(
// cusolverDnSormtr-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnSormtr-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnSormtr-NEXT:       n /*int*/, (float*)a /*float **/, lda /*int*/, (float*)tau /*float **/, (float*)c /*float **/,
// cusolverDnSormtr-NEXT:       ldc /*int*/, (float*)buffer /*float **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDormtr | FileCheck %s -check-prefix=cusolverDnDormtr
// cusolverDnDormtr: CUDA API:
// cusolverDnDormtr-NEXT:   cusolverDnDormtr(
// cusolverDnDormtr-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnDormtr-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnDormtr-NEXT:       n /*int*/, a /*double **/, lda /*int*/, tau /*double **/, c /*double **/,
// cusolverDnDormtr-NEXT:       ldc /*int*/, buffer /*double **/, buffer_size /*int*/, info /*int **/);
// cusolverDnDormtr-NEXT: Is migrated to:
// cusolverDnDormtr-NEXT:   oneapi::mkl::lapack::ormtr(
// cusolverDnDormtr-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnDormtr-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnDormtr-NEXT:       n /*int*/, (double*)a /*double **/, lda /*int*/, (double*)tau /*double **/, (double*)c /*double **/,
// cusolverDnDormtr-NEXT:       ldc /*int*/, (double*)buffer /*double **/, buffer_size /*int **/);
