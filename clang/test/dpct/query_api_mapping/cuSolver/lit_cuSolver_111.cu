// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnGesvd | FileCheck %s -check-prefix=cusolverDnGesvd
// cusolverDnGesvd: CUDA API:
// cusolverDnGesvd-NEXT:   cusolverDnGesvd(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnGesvd-NEXT:                   jobu /*signed char*/, jobvt /*signed char*/, m /*int64_t*/,
// cusolverDnGesvd-NEXT:                   n /*int64_t*/, a_type /*cudaDataType*/, a /*void **/,
// cusolverDnGesvd-NEXT:                   lda /*int64_t*/, s_type /*cudaDataType*/, s /*void **/,
// cusolverDnGesvd-NEXT:                   u_type /*cudaDataType*/, u /*void **/, ldu /*int64_t*/,
// cusolverDnGesvd-NEXT:                   vt_type /*cudaDataType*/, vt /*void **/, ldvt /*int64_t*/,
// cusolverDnGesvd-NEXT:                   compute_type /*cudaDataType*/, buffer /*void **/,
// cusolverDnGesvd-NEXT:                   buffer_size /*size_t*/, info /*int **/);
// cusolverDnGesvd-NEXT: Is migrated to:
// cusolverDnGesvd-NEXT:   dpct::lapack::gesvd(*handle, jobu, jobvt, m, n, a_type, a, lda, s_type, s, u_type, u, ldu, vt_type, vt, ldvt, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnGesvd_bufferSize | FileCheck %s -check-prefix=cusolverDnGesvd_bufferSize
// cusolverDnGesvd_bufferSize: CUDA API:
// cusolverDnGesvd_bufferSize-NEXT:   size_t buffer_size;
// cusolverDnGesvd_bufferSize-NEXT:   cusolverDnGesvd_bufferSize(
// cusolverDnGesvd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnGesvd_bufferSize-NEXT:       jobu /*signed char*/, jobvt /*signed char*/, m /*int64_t*/, n /*int64_t*/,
// cusolverDnGesvd_bufferSize-NEXT:       a_type /*cudaDataType*/, a /*const void **/, lda /*int64_t*/,
// cusolverDnGesvd_bufferSize-NEXT:       s_type /*cudaDataType*/, s /*const void **/, u_type /*cudaDataType*/,
// cusolverDnGesvd_bufferSize-NEXT:       u /*const void **/, ldu /*int64_t*/, vt_type /*cudaDataType*/,
// cusolverDnGesvd_bufferSize-NEXT:       vt /*const void **/, ldvt /*int64_t*/, compute_type /*cudaDataType*/,
// cusolverDnGesvd_bufferSize-NEXT:       &buffer_size /*size_t **/);
// cusolverDnGesvd_bufferSize-NEXT: Is migrated to:
// cusolverDnGesvd_bufferSize-NEXT:   size_t buffer_size;
// cusolverDnGesvd_bufferSize-NEXT:   dpct::lapack::gesvd_scratchpad_size(*handle, jobu, jobvt, m, n, a_type, lda, u_type, ldu, vt_type, ldvt, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXgeqrf | FileCheck %s -check-prefix=cusolverDnXgeqrf
// cusolverDnXgeqrf: CUDA API:
// cusolverDnXgeqrf-NEXT:   cusolverDnXgeqrf(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXgeqrf-NEXT:                    m /*int64_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
// cusolverDnXgeqrf-NEXT:                    a /*void **/, lda /*int64_t*/, tau_type /*cudaDataType*/,
// cusolverDnXgeqrf-NEXT:                    tau /*void **/, compute_type /*cudaDataType*/,
// cusolverDnXgeqrf-NEXT:                    device_buffer /*void **/, device_buffer_size /*size_t*/,
// cusolverDnXgeqrf-NEXT:                    host_buffer /*void **/, host_buffer_size /*size_t*/,
// cusolverDnXgeqrf-NEXT:                    info /*int **/);
// cusolverDnXgeqrf-NEXT: Is migrated to:
// cusolverDnXgeqrf-NEXT:   dpct::lapack::geqrf(*handle, m, n, a_type, a, lda, tau_type, tau, device_buffer, device_buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXgeqrf_bufferSize | FileCheck %s -check-prefix=cusolverDnXgeqrf_bufferSize
// cusolverDnXgeqrf_bufferSize: CUDA API:
// cusolverDnXgeqrf_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXgeqrf_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXgeqrf_bufferSize-NEXT:   cusolverDnXgeqrf_bufferSize(
// cusolverDnXgeqrf_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXgeqrf_bufferSize-NEXT:       m /*int64_t*/, n /*int64_t*/, a_type /*cudaDataType*/, a /*const void **/,
// cusolverDnXgeqrf_bufferSize-NEXT:       lda /*int64_t*/, tau_type /*cudaDataType*/, tau /*const void **/,
// cusolverDnXgeqrf_bufferSize-NEXT:       compute_type /*cudaDataType*/, &device_buffer_size /*size_t **/,
// cusolverDnXgeqrf_bufferSize-NEXT:       &host_buffer_size /*size_t **/);
// cusolverDnXgeqrf_bufferSize-NEXT: Is migrated to:
// cusolverDnXgeqrf_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXgeqrf_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXgeqrf_bufferSize-NEXT:   dpct::lapack::geqrf_scratchpad_size(*handle, m, n, a_type, lda, &device_buffer_size, &host_buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXgesvd | FileCheck %s -check-prefix=cusolverDnXgesvd
// cusolverDnXgesvd: CUDA API:
// cusolverDnXgesvd-NEXT:   cusolverDnXgesvd(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXgesvd-NEXT:                    jobu /*signed char*/, jobvt /*signed char*/, m /*int64_t*/,
// cusolverDnXgesvd-NEXT:                    n /*int64_t*/, a_type /*cudaDataType*/, a /*void **/,
// cusolverDnXgesvd-NEXT:                    lda /*int64_t*/, s_type /*cudaDataType*/, s /*void **/,
// cusolverDnXgesvd-NEXT:                    u_type /*cudaDataType*/, u /*void **/, ldu /*int64_t*/,
// cusolverDnXgesvd-NEXT:                    vt_type /*cudaDataType*/, vt /*void **/, ldvt /*int64_t*/,
// cusolverDnXgesvd-NEXT:                    compute_type /*cudaDataType*/, device_buffer /*void **/,
// cusolverDnXgesvd-NEXT:                    device_buffer_size /*size_t*/, host_buffer /*void **/,
// cusolverDnXgesvd-NEXT:                    host_buffer_size /*size_t*/, info /*int **/);
// cusolverDnXgesvd-NEXT: Is migrated to:
// cusolverDnXgesvd-NEXT:   dpct::lapack::gesvd(*handle, jobu, jobvt, m, n, a_type, a, lda, s_type, s, u_type, u, ldu, vt_type, vt, ldvt, device_buffer, device_buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXgesvd_bufferSize | FileCheck %s -check-prefix=cusolverDnXgesvd_bufferSize
// cusolverDnXgesvd_bufferSize: CUDA API:
// cusolverDnXgesvd_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXgesvd_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXgesvd_bufferSize-NEXT:   cusolverDnXgesvd_bufferSize(
// cusolverDnXgesvd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXgesvd_bufferSize-NEXT:       jobu /*signed char*/, jobvt /*signed char*/, m /*int64_t*/, n /*int64_t*/,
// cusolverDnXgesvd_bufferSize-NEXT:       a_type /*cudaDataType*/, a /*const void **/, lda /*int64_t*/,
// cusolverDnXgesvd_bufferSize-NEXT:       s_type /*cudaDataType*/, s /*const void **/, u_type /*cudaDataType*/,
// cusolverDnXgesvd_bufferSize-NEXT:       u /*const void **/, ldu /*int64_t*/, vt_type /*cudaDataType*/,
// cusolverDnXgesvd_bufferSize-NEXT:       vt /*const void **/, ldvt /*int64_t*/, compute_type /*cudaDataType*/,
// cusolverDnXgesvd_bufferSize-NEXT:       &device_buffer_size /*size_t **/, &host_buffer_size /*size_t **/);
// cusolverDnXgesvd_bufferSize-NEXT: Is migrated to:
// cusolverDnXgesvd_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXgesvd_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXgesvd_bufferSize-NEXT:   dpct::lapack::gesvd_scratchpad_size(*handle, jobu, jobvt, m, n, a_type, lda, u_type, ldu, vt_type, ldvt, &device_buffer_size, &host_buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXgetrf | FileCheck %s -check-prefix=cusolverDnXgetrf
// cusolverDnXgetrf: CUDA API:
// cusolverDnXgetrf-NEXT:   cusolverDnXgetrf(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXgetrf-NEXT:                    m /*int64_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
// cusolverDnXgetrf-NEXT:                    a /*void **/, lda /*int64_t*/, ipiv /*int64_t **/,
// cusolverDnXgetrf-NEXT:                    compute_type /*cudaDataType*/, device_buffer /*void **/,
// cusolverDnXgetrf-NEXT:                    device_buffer_size /*size_t*/, host_buffer /*void **/,
// cusolverDnXgetrf-NEXT:                    host_buffer_size /*size_t*/, info /*int **/);
// cusolverDnXgetrf-NEXT: Is migrated to:
// cusolverDnXgetrf-NEXT:   dpct::lapack::getrf(*handle, m, n, a_type, a, lda, ipiv, device_buffer, device_buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXgetrf_bufferSize | FileCheck %s -check-prefix=cusolverDnXgetrf_bufferSize
// cusolverDnXgetrf_bufferSize: CUDA API:
// cusolverDnXgetrf_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXgetrf_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXgetrf_bufferSize-NEXT:   cusolverDnXgetrf_bufferSize(
// cusolverDnXgetrf_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXgetrf_bufferSize-NEXT:       m /*int64_t*/, n /*int64_t*/, a_type /*cudaDataType*/, a /*const void **/,
// cusolverDnXgetrf_bufferSize-NEXT:       lda /*int64_t*/, compute_type /*cudaDataType*/,
// cusolverDnXgetrf_bufferSize-NEXT:       &device_buffer_size /*size_t **/, &host_buffer_size /*size_t **/);
// cusolverDnXgetrf_bufferSize-NEXT: Is migrated to:
// cusolverDnXgetrf_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXgetrf_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXgetrf_bufferSize-NEXT:   dpct::lapack::getrf_scratchpad_size(*handle, m, n, a_type, lda, &device_buffer_size, &host_buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXgetrs | FileCheck %s -check-prefix=cusolverDnXgetrs
// cusolverDnXgetrs: CUDA API:
// cusolverDnXgetrs-NEXT:   cusolverDnXgetrs(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXgetrs-NEXT:                    trans /*cublasOperation_t*/, n /*int64_t*/, nrhs /*int64_t*/,
// cusolverDnXgetrs-NEXT:                    a_type /*cudaDataType*/, a /*const void **/, lda /*int64_t*/,
// cusolverDnXgetrs-NEXT:                    ipiv /*const int64_t **/, b_type /*cudaDataType*/,
// cusolverDnXgetrs-NEXT:                    b /*void **/, ldb /*int64_t*/, info /*int **/);
// cusolverDnXgetrs-NEXT: Is migrated to:
// cusolverDnXgetrs-NEXT:   dpct::lapack::getrs(*handle, trans, n, nrhs, a_type, a, lda, ipiv, b_type, b, ldb, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXpotrf | FileCheck %s -check-prefix=cusolverDnXpotrf
// cusolverDnXpotrf: CUDA API:
// cusolverDnXpotrf-NEXT:   cusolverDnXpotrf(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXpotrf-NEXT:                    uplo /*cublasFillMode_t*/, n /*int64_t*/,
// cusolverDnXpotrf-NEXT:                    a_type /*cudaDataType*/, a /*void **/, lda /*int64_t*/,
// cusolverDnXpotrf-NEXT:                    compute_type /*cudaDataType*/, device_buffer /*void **/,
// cusolverDnXpotrf-NEXT:                    device_buffer_size /*size_t*/, host_buffer /*void **/,
// cusolverDnXpotrf-NEXT:                    host_buffer_size /*size_t*/, info /*int **/);
// cusolverDnXpotrf-NEXT: Is migrated to:
// cusolverDnXpotrf-NEXT:   dpct::lapack::potrf(*handle, uplo, n, a_type, a, lda, device_buffer, device_buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXpotrf_bufferSize | FileCheck %s -check-prefix=cusolverDnXpotrf_bufferSize
// cusolverDnXpotrf_bufferSize: CUDA API:
// cusolverDnXpotrf_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXpotrf_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXpotrf_bufferSize-NEXT:   cusolverDnXpotrf_bufferSize(
// cusolverDnXpotrf_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXpotrf_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
// cusolverDnXpotrf_bufferSize-NEXT:       a /*const void **/, lda /*int64_t*/, compute_type /*cudaDataType*/,
// cusolverDnXpotrf_bufferSize-NEXT:       &device_buffer_size /*size_t **/, &host_buffer_size /*size_t **/);
// cusolverDnXpotrf_bufferSize-NEXT: Is migrated to:
// cusolverDnXpotrf_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXpotrf_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXpotrf_bufferSize-NEXT:   dpct::lapack::potrf_scratchpad_size(*handle, uplo, n, a_type, lda, &device_buffer_size, &host_buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXpotrs | FileCheck %s -check-prefix=cusolverDnXpotrs
// cusolverDnXpotrs: CUDA API:
// cusolverDnXpotrs-NEXT:   cusolverDnXpotrs(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXpotrs-NEXT:                    uplo /*cublasFillMode_t*/, n /*int64_t*/, nrhs /*int64_t*/,
// cusolverDnXpotrs-NEXT:                    a_type /*cudaDataType*/, a /*const void **/, lda /*int64_t*/,
// cusolverDnXpotrs-NEXT:                    b_type /*cudaDataType*/, b /*void **/, ldb /*int64_t*/,
// cusolverDnXpotrs-NEXT:                    info /*int **/);
// cusolverDnXpotrs-NEXT: Is migrated to:
// cusolverDnXpotrs-NEXT:   dpct::lapack::potrs(*handle, uplo, n, nrhs, a_type, a, lda, b_type, b, ldb, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXsyevd | FileCheck %s -check-prefix=cusolverDnXsyevd
// cusolverDnXsyevd: CUDA API:
// cusolverDnXsyevd-NEXT:   cusolverDnXsyevd(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXsyevd-NEXT:                    jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnXsyevd-NEXT:                    n /*int64_t*/, a_type /*cudaDataType*/, a /*void **/,
// cusolverDnXsyevd-NEXT:                    lda /*int64_t*/, w_type /*cudaDataType*/, w /*void **/,
// cusolverDnXsyevd-NEXT:                    compute_type /*cudaDataType*/, device_buffer /*void **/,
// cusolverDnXsyevd-NEXT:                    device_buffer_size /*size_t*/, host_buffer /*void **/,
// cusolverDnXsyevd-NEXT:                    host_buffer_size /*size_t*/, info /*int **/);
// cusolverDnXsyevd-NEXT: Is migrated to:
// cusolverDnXsyevd-NEXT:   dpct::lapack::syheevd(*handle, jobz, uplo, n, a_type, a, lda, w_type, w, device_buffer, device_buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXsyevd_bufferSize | FileCheck %s -check-prefix=cusolverDnXsyevd_bufferSize
// cusolverDnXsyevd_bufferSize: CUDA API:
// cusolverDnXsyevd_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXsyevd_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXsyevd_bufferSize-NEXT:   cusolverDnXsyevd_bufferSize(
// cusolverDnXsyevd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXsyevd_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int64_t*/,
// cusolverDnXsyevd_bufferSize-NEXT:       a_type /*cudaDataType*/, a /*const void **/, lda /*int64_t*/,
// cusolverDnXsyevd_bufferSize-NEXT:       w_type /*cudaDataType*/, w /*const void **/,
// cusolverDnXsyevd_bufferSize-NEXT:       compute_type /*cudaDataType*/, &device_buffer_size /*size_t **/,
// cusolverDnXsyevd_bufferSize-NEXT:       &host_buffer_size /*size_t **/);
// cusolverDnXsyevd_bufferSize-NEXT: Is migrated to:
// cusolverDnXsyevd_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXsyevd_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXsyevd_bufferSize-NEXT:   dpct::lapack::syheevd_scratchpad_size(*handle, jobz, uplo, n, a_type, lda, w_type, &device_buffer_size, &host_buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXsyevdx | FileCheck %s -check-prefix=cusolverDnXsyevdx
// cusolverDnXsyevdx: CUDA API:
// cusolverDnXsyevdx-NEXT:   cusolverDnXsyevdx(
// cusolverDnXsyevdx-NEXT:       handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXsyevdx-NEXT:       jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
// cusolverDnXsyevdx-NEXT:       uplo /*cublasFillMode_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
// cusolverDnXsyevdx-NEXT:       a /*void **/, lda /*int64_t*/, vl /*void **/, vu /*void **/,
// cusolverDnXsyevdx-NEXT:       il /*int64_t*/, iu /*int64_t*/, h_meig /*int64_t **/,
// cusolverDnXsyevdx-NEXT:       w_type /*cudaDataType*/, w /*void **/, compute_type /*cudaDataType*/,
// cusolverDnXsyevdx-NEXT:       device_buffer /*void **/, device_buffer_size /*size_t*/,
// cusolverDnXsyevdx-NEXT:       host_buffer /*void **/, host_buffer_size /*size_t*/, info /*int **/);
// cusolverDnXsyevdx-NEXT: Is migrated to:
// cusolverDnXsyevdx-NEXT:   dpct::lapack::syheevx(*handle, jobz, range, uplo, n, a_type, a, lda, vl, vu, il, iu, h_meig, w_type, w, device_buffer, device_buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXsyevdx_bufferSize | FileCheck %s -check-prefix=cusolverDnXsyevdx_bufferSize
// cusolverDnXsyevdx_bufferSize: CUDA API:
// cusolverDnXsyevdx_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXsyevdx_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXsyevdx_bufferSize-NEXT:   cusolverDnXsyevdx_bufferSize(
// cusolverDnXsyevdx_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXsyevdx_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
// cusolverDnXsyevdx_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
// cusolverDnXsyevdx_bufferSize-NEXT:       a /*const void **/, lda /*int64_t*/, vl /*void **/, vu /*void **/,
// cusolverDnXsyevdx_bufferSize-NEXT:       il /*int64_t*/, iu /*int64_t*/, h_meig /*int64_t **/,
// cusolverDnXsyevdx_bufferSize-NEXT:       w_type /*cudaDataType*/, w /*const void **/,
// cusolverDnXsyevdx_bufferSize-NEXT:       compute_type /*cudaDataType*/, &device_buffer_size /*size_t **/,
// cusolverDnXsyevdx_bufferSize-NEXT:       &host_buffer_size /*size_t **/);
// cusolverDnXsyevdx_bufferSize-NEXT: Is migrated to:
// cusolverDnXsyevdx_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXsyevdx_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXsyevdx_bufferSize-NEXT:   dpct::lapack::syheevx_scratchpad_size(*handle, jobz, range, uplo, n, a_type, lda, vl, vu, il, iu, w_type, &device_buffer_size, &host_buffer_size);
