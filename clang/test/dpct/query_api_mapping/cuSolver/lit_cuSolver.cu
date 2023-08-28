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
// cusolverDnSpotrsBatched-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSpotrfBatched | FileCheck %s -check-prefix=cusolverDnSpotrfBatched
// cusolverDnSpotrfBatched: CUDA API:
// cusolverDnSpotrfBatched-NEXT:   cusolverDnSpotrfBatched(handle /*cusolverDnHandle_t*/,
// cusolverDnSpotrfBatched-NEXT:                           upper_lower /*cublasFillMode_t*/, n /*int*/,
// cusolverDnSpotrfBatched-NEXT:                           a /*float ***/, lda /*int*/, info /*int **/,
// cusolverDnSpotrfBatched-NEXT:                           group_count /*int*/);
// cusolverDnSpotrfBatched-NEXT: Is migrated to:
// cusolverDnSpotrfBatched-NEXT:   dpct::lapack::potrf_batch(*handle, upper_lower, n, a, lda, info, group_count);
// cusolverDnSpotrfBatched-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDpotrfBatched | FileCheck %s -check-prefix=cusolverDnDpotrfBatched
// cusolverDnDpotrfBatched: CUDA API:
// cusolverDnDpotrfBatched-NEXT:   cusolverDnDpotrfBatched(handle /*cusolverDnHandle_t*/,
// cusolverDnDpotrfBatched-NEXT:                           upper_lower /*cublasFillMode_t*/, n /*int*/,
// cusolverDnDpotrfBatched-NEXT:                           a /*double ***/, lda /*int*/, info /*int **/,
// cusolverDnDpotrfBatched-NEXT:                           group_count /*int*/);
// cusolverDnDpotrfBatched-NEXT: Is migrated to:
// cusolverDnDpotrfBatched-NEXT:   dpct::lapack::potrf_batch(*handle, upper_lower, n, a, lda, info, group_count);
// cusolverDnDpotrfBatched-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCpotrsBatched | FileCheck %s -check-prefix=cusolverDnCpotrsBatched
// cusolverDnCpotrsBatched: CUDA API:
// cusolverDnCpotrsBatched-NEXT:   cusolverDnCpotrsBatched(
// cusolverDnCpotrsBatched-NEXT:       handle /*cusolverDnHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cusolverDnCpotrsBatched-NEXT:       n /*int*/, nrhs /*int*/, a /*cuComplex ***/, lda /*int*/,
// cusolverDnCpotrsBatched-NEXT:       b /*cuComplex ***/, ldb /*int*/, info /*int **/, group_count /*int*/);
// cusolverDnCpotrsBatched-NEXT: Is migrated to:
// cusolverDnCpotrsBatched-NEXT:   dpct::lapack::potrs_batch(*handle, upper_lower, n, nrhs, a, lda, b, ldb, info, group_count);
// cusolverDnCpotrsBatched-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZpotrsBatched | FileCheck %s -check-prefix=cusolverDnZpotrsBatched
// cusolverDnZpotrsBatched: CUDA API:
// cusolverDnZpotrsBatched-NEXT:   cusolverDnZpotrsBatched(handle /*cusolverDnHandle_t*/,
// cusolverDnZpotrsBatched-NEXT:                           upper_lower /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZpotrsBatched-NEXT:                           nrhs /*int*/, a /*cuDoubleComplex ***/, lda /*int*/,
// cusolverDnZpotrsBatched-NEXT:                           b /*cuDoubleComplex ***/, ldb /*int*/, info /*int **/,
// cusolverDnZpotrsBatched-NEXT:                           group_count /*int*/);
// cusolverDnZpotrsBatched-NEXT: Is migrated to:
// cusolverDnZpotrsBatched-NEXT:   dpct::lapack::potrs_batch(*handle, upper_lower, n, nrhs, a, lda, b, ldb, info, group_count);
// cusolverDnZpotrsBatched-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDpotrsBatched | FileCheck %s -check-prefix=cusolverDnDpotrsBatched
// cusolverDnDpotrsBatched: CUDA API:
// cusolverDnDpotrsBatched-NEXT:   cusolverDnDpotrsBatched(
// cusolverDnDpotrsBatched-NEXT:       handle /*cusolverDnHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cusolverDnDpotrsBatched-NEXT:       n /*int*/, nrhs /*int*/, a /*double ***/, lda /*int*/, b /*double ***/,
// cusolverDnDpotrsBatched-NEXT:       ldb /*int*/, info /*int **/, group_count /*int*/);
// cusolverDnDpotrsBatched-NEXT: Is migrated to:
// cusolverDnDpotrsBatched-NEXT:   dpct::lapack::potrs_batch(*handle, upper_lower, n, nrhs, a, lda, b, ldb, info, group_count);
// cusolverDnDpotrsBatched-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZpotrfBatched | FileCheck %s -check-prefix=cusolverDnZpotrfBatched
// cusolverDnZpotrfBatched: CUDA API:
// cusolverDnZpotrfBatched-NEXT:   cusolverDnZpotrfBatched(handle /*cusolverDnHandle_t*/,
// cusolverDnZpotrfBatched-NEXT:                           upper_lower /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZpotrfBatched-NEXT:                           a /*cuDoubleComplex ***/, lda /*int*/, info /*int **/,
// cusolverDnZpotrfBatched-NEXT:                           group_count /*int*/);
// cusolverDnZpotrfBatched-NEXT: Is migrated to:
// cusolverDnZpotrfBatched-NEXT:   dpct::lapack::potrf_batch(*handle, upper_lower, n, a, lda, info, group_count);
// cusolverDnZpotrfBatched-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCpotrfBatched | FileCheck %s -check-prefix=cusolverDnCpotrfBatched
// cusolverDnCpotrfBatched: CUDA API:
// cusolverDnCpotrfBatched-NEXT:   cusolverDnCpotrfBatched(handle /*cusolverDnHandle_t*/,
// cusolverDnCpotrfBatched-NEXT:                           upper_lower /*cublasFillMode_t*/, n /*int*/,
// cusolverDnCpotrfBatched-NEXT:                           a /*cuComplex ***/, lda /*int*/, info /*int **/,
// cusolverDnCpotrfBatched-NEXT:                           group_count /*int*/);
// cusolverDnCpotrfBatched-NEXT: Is migrated to:
// cusolverDnCpotrfBatched-NEXT:   dpct::lapack::potrf_batch(*handle, upper_lower, n, a, lda, info, group_count);
// cusolverDnCpotrfBatched-EMPTY:

