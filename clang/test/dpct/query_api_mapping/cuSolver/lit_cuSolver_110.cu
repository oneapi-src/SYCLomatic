// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCreateParams | FileCheck %s -check-prefix=cusolverDnCreateParams
// cusolverDnCreateParams: CUDA API:
// cusolverDnCreateParams-NEXT:   cusolverDnCreateParams(params /*cusolverDnParams_t **/);
// cusolverDnCreateParams-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDestroyParams | FileCheck %s -check-prefix=cusolverDnDestroyParams
// cusolverDnDestroyParams: CUDA API:
// cusolverDnDestroyParams-NEXT:   cusolverDnDestroyParams(params /*cusolverDnParams_t*/);
// cusolverDnDestroyParams-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnGeqrf | FileCheck %s -check-prefix=cusolverDnGeqrf
// cusolverDnGeqrf: CUDA API:
// cusolverDnGeqrf-NEXT:   cusolverDnGeqrf(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnGeqrf-NEXT:                   m /*int64_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
// cusolverDnGeqrf-NEXT:                   a /*void **/, lda /*int64_t*/, tau_type /*cudaDataType*/,
// cusolverDnGeqrf-NEXT:                   tau /*void **/, compute_type /*cudaDataType*/,
// cusolverDnGeqrf-NEXT:                   buffer /*void **/, buffer_size /*size_t*/, info /*int **/);
// cusolverDnGeqrf-NEXT: Is migrated to:
// cusolverDnGeqrf-NEXT:   dpct::lapack::geqrf(*handle, m, n, a_type, a, lda, tau_type, tau, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnGeqrf_bufferSize | FileCheck %s -check-prefix=cusolverDnGeqrf_bufferSize
// cusolverDnGeqrf_bufferSize: CUDA API:
// cusolverDnGeqrf_bufferSize-NEXT:   size_t buffer_size;
// cusolverDnGeqrf_bufferSize-NEXT:   cusolverDnGeqrf_bufferSize(
// cusolverDnGeqrf_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnGeqrf_bufferSize-NEXT:       m /*int64_t*/, n /*int64_t*/, a_type /*cudaDataType*/, a /*const void **/,
// cusolverDnGeqrf_bufferSize-NEXT:       lda /*int64_t*/, tau_type /*cudaDataType*/, tau /*const void **/,
// cusolverDnGeqrf_bufferSize-NEXT:       compute_type /*cudaDataType*/, &buffer_size /*size_t **/);
// cusolverDnGeqrf_bufferSize-NEXT: Is migrated to:
// cusolverDnGeqrf_bufferSize-NEXT:   size_t buffer_size;
// cusolverDnGeqrf_bufferSize-NEXT:   dpct::lapack::geqrf_scratchpad_size(*handle, m, n, a_type, lda, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnGetrf | FileCheck %s -check-prefix=cusolverDnGetrf
// cusolverDnGetrf: CUDA API:
// cusolverDnGetrf-NEXT:   cusolverDnGetrf(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnGetrf-NEXT:                   m /*int64_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
// cusolverDnGetrf-NEXT:                   a /*void **/, lda /*int64_t*/, ipiv /*int64_t **/,
// cusolverDnGetrf-NEXT:                   compute_type /*cudaDataType*/, buffer /*void **/,
// cusolverDnGetrf-NEXT:                   buffer_size /*size_t*/, info /*int **/);
// cusolverDnGetrf-NEXT: Is migrated to:
// cusolverDnGetrf-NEXT:   dpct::lapack::getrf(*handle, m, n, a_type, a, lda, ipiv, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnGetrf_bufferSize | FileCheck %s -check-prefix=cusolverDnGetrf_bufferSize
// cusolverDnGetrf_bufferSize: CUDA API:
// cusolverDnGetrf_bufferSize-NEXT:   size_t buffer_size;
// cusolverDnGetrf_bufferSize-NEXT:   cusolverDnGetrf_bufferSize(
// cusolverDnGetrf_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnGetrf_bufferSize-NEXT:       m /*int64_t*/, n /*int64_t*/, a_type /*cudaDataType*/, a /*const void **/,
// cusolverDnGetrf_bufferSize-NEXT:       lda /*int64_t*/, compute_type /*cudaDataType*/,
// cusolverDnGetrf_bufferSize-NEXT:       &buffer_size /*size_t **/);
// cusolverDnGetrf_bufferSize-NEXT: Is migrated to:
// cusolverDnGetrf_bufferSize-NEXT:   size_t buffer_size;
// cusolverDnGetrf_bufferSize-NEXT:   dpct::lapack::getrf_scratchpad_size(*handle, m, n, a_type, lda, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnGetrs | FileCheck %s -check-prefix=cusolverDnGetrs
// cusolverDnGetrs: CUDA API:
// cusolverDnGetrs-NEXT:   cusolverDnGetrs(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnGetrs-NEXT:                   trans /*cublasOperation_t*/, n /*int64_t*/, nrhs /*int64_t*/,
// cusolverDnGetrs-NEXT:                   a_type /*cudaDataType*/, a /*const void **/, lda /*int64_t*/,
// cusolverDnGetrs-NEXT:                   ipiv /*const int64_t **/, b_type /*cudaDataType*/,
// cusolverDnGetrs-NEXT:                   b /*void **/, ldb /*int64_t*/, info /*int **/);
// cusolverDnGetrs-NEXT: Is migrated to:
// cusolverDnGetrs-NEXT:   dpct::lapack::getrs(*handle, trans, n, nrhs, a_type, a, lda, ipiv, b_type, b, ldb, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnPotrf | FileCheck %s -check-prefix=cusolverDnPotrf
// cusolverDnPotrf: CUDA API:
// cusolverDnPotrf-NEXT:   cusolverDnPotrf(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnPotrf-NEXT:                   uplo /*cublasFillMode_t*/, n /*int64_t*/,
// cusolverDnPotrf-NEXT:                   a_type /*cudaDataType*/, a /*void **/, lda /*int64_t*/,
// cusolverDnPotrf-NEXT:                   compute_type /*cudaDataType*/, buffer /*void **/,
// cusolverDnPotrf-NEXT:                   buffer_size /*size_t*/, info /*int **/);
// cusolverDnPotrf-NEXT: Is migrated to:
// cusolverDnPotrf-NEXT:   dpct::lapack::potrf(*handle, uplo, n, a_type, a, lda, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnPotrf_bufferSize | FileCheck %s -check-prefix=cusolverDnPotrf_bufferSize
// cusolverDnPotrf_bufferSize: CUDA API:
// cusolverDnPotrf_bufferSize-NEXT:   size_t buffer_size;
// cusolverDnPotrf_bufferSize-NEXT:   cusolverDnPotrf_bufferSize(
// cusolverDnPotrf_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnPotrf_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
// cusolverDnPotrf_bufferSize-NEXT:       a /*const void **/, lda /*int64_t*/, compute_type /*cudaDataType*/,
// cusolverDnPotrf_bufferSize-NEXT:       &buffer_size /*size_t **/);
// cusolverDnPotrf_bufferSize-NEXT: Is migrated to:
// cusolverDnPotrf_bufferSize-NEXT:   size_t buffer_size;
// cusolverDnPotrf_bufferSize-NEXT:   dpct::lapack::potrf_scratchpad_size(*handle, uplo, n, a_type, lda, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnPotrs | FileCheck %s -check-prefix=cusolverDnPotrs
// cusolverDnPotrs: CUDA API:
// cusolverDnPotrs-NEXT:   cusolverDnPotrs(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnPotrs-NEXT:                   uplo /*cublasFillMode_t*/, n /*int64_t*/, nrhs /*int64_t*/,
// cusolverDnPotrs-NEXT:                   a_type /*cudaDataType*/, a /*const void **/, lda /*int64_t*/,
// cusolverDnPotrs-NEXT:                   b_type /*cudaDataType*/, b /*void **/, ldb /*int64_t*/,
// cusolverDnPotrs-NEXT:                   info /*int **/);
// cusolverDnPotrs-NEXT: Is migrated to:
// cusolverDnPotrs-NEXT:   dpct::lapack::potrs(*handle, uplo, n, nrhs, a_type, a, lda, b_type, b, ldb, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSetAdvOptions | FileCheck %s -check-prefix=cusolverDnSetAdvOptions
// cusolverDnSetAdvOptions: CUDA API:
// cusolverDnSetAdvOptions-NEXT:   cusolverDnSetAdvOptions(params /*cusolverDnParams_t*/,
// cusolverDnSetAdvOptions-NEXT:                           func /*cusolverDnFunction_t*/,
// cusolverDnSetAdvOptions-NEXT:                           algo /*cusolverAlgMode_t*/);
// cusolverDnSetAdvOptions-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSyevd | FileCheck %s -check-prefix=cusolverDnSyevd
// cusolverDnSyevd: CUDA API:
// cusolverDnSyevd-NEXT:   cusolverDnSyevd(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnSyevd-NEXT:                   jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSyevd-NEXT:                   n /*int64_t*/, a_type /*cudaDataType*/, a /*void **/,
// cusolverDnSyevd-NEXT:                   lda /*int64_t*/, w_type /*cudaDataType*/, w /*void **/,
// cusolverDnSyevd-NEXT:                   compute_type /*cudaDataType*/, buffer /*void **/,
// cusolverDnSyevd-NEXT:                   buffer_size /*size_t*/, info /*int **/);
// cusolverDnSyevd-NEXT: Is migrated to:
// cusolverDnSyevd-NEXT:   dpct::lapack::syheevd(*handle, jobz, uplo, n, a_type, a, lda, w_type, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSyevd_bufferSize | FileCheck %s -check-prefix=cusolverDnSyevd_bufferSize
// cusolverDnSyevd_bufferSize: CUDA API:
// cusolverDnSyevd_bufferSize-NEXT:   size_t buffer_size;
// cusolverDnSyevd_bufferSize-NEXT:   cusolverDnSyevd_bufferSize(
// cusolverDnSyevd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnSyevd_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int64_t*/,
// cusolverDnSyevd_bufferSize-NEXT:       a_type /*cudaDataType*/, a /*const void **/, lda /*int64_t*/,
// cusolverDnSyevd_bufferSize-NEXT:       w_type /*cudaDataType*/, w /*const void **/,
// cusolverDnSyevd_bufferSize-NEXT:       compute_type /*cudaDataType*/, &buffer_size /*size_t **/);
// cusolverDnSyevd_bufferSize-NEXT: Is migrated to:
// cusolverDnSyevd_bufferSize-NEXT:   size_t buffer_size;
// cusolverDnSyevd_bufferSize-NEXT:   dpct::lapack::syheevd_scratchpad_size(*handle, jobz, uplo, n, a_type, lda, w_type, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSyevdx | FileCheck %s -check-prefix=cusolverDnSyevdx
// cusolverDnSyevdx: CUDA API:
// cusolverDnSyevdx-NEXT:   cusolverDnSyevdx(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnSyevdx-NEXT:                    jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
// cusolverDnSyevdx-NEXT:                    uplo /*cublasFillMode_t*/, n /*int64_t*/,
// cusolverDnSyevdx-NEXT:                    a_type /*cudaDataType*/, a /*void **/, lda /*int64_t*/,
// cusolverDnSyevdx-NEXT:                    vl /*void **/, vu /*void **/, il /*int64_t*/, iu /*int64_t*/,
// cusolverDnSyevdx-NEXT:                    h_meig /*int64_t **/, w_type /*cudaDataType*/, w /*void **/,
// cusolverDnSyevdx-NEXT:                    compute_type /*cudaDataType*/, buffer /*void **/,
// cusolverDnSyevdx-NEXT:                    buffer_size /*size_t*/, info /*int **/);
// cusolverDnSyevdx-NEXT: Is migrated to:
// cusolverDnSyevdx-NEXT:   dpct::lapack::syheevx(*handle, jobz, range, uplo, n, a_type, a, lda, vl, vu, il, iu, h_meig, w_type, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSyevdx_bufferSize | FileCheck %s -check-prefix=cusolverDnSyevdx_bufferSize
// cusolverDnSyevdx_bufferSize: CUDA API:
// cusolverDnSyevdx_bufferSize-NEXT:   size_t buffer_size;
// cusolverDnSyevdx_bufferSize-NEXT:   cusolverDnSyevdx_bufferSize(
// cusolverDnSyevdx_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnSyevdx_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
// cusolverDnSyevdx_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
// cusolverDnSyevdx_bufferSize-NEXT:       a /*const void **/, lda /*int64_t*/, vl /*void **/, vu /*void **/,
// cusolverDnSyevdx_bufferSize-NEXT:       il /*int64_t*/, iu /*int64_t*/, h_meig /*int64_t **/,
// cusolverDnSyevdx_bufferSize-NEXT:       w_type /*cudaDataType*/, w /*const void **/,
// cusolverDnSyevdx_bufferSize-NEXT:       compute_type /*cudaDataType*/, &buffer_size /*size_t **/);
// cusolverDnSyevdx_bufferSize-NEXT: Is migrated to:
// cusolverDnSyevdx_bufferSize-NEXT:   size_t buffer_size;
// cusolverDnSyevdx_bufferSize-NEXT:   dpct::lapack::syheevx_scratchpad_size(*handle, jobz, range, uplo, n, a_type, lda, vl, vu, il, iu, w_type, &buffer_size);
