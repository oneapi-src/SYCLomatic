// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXtrtri | FileCheck %s -check-prefix=cusolverDnXtrtri
// cusolverDnXtrtri: CUDA API:
// cusolverDnXtrtri-NEXT:   cusolverDnXtrtri(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnXtrtri-NEXT:                    diag /*cublasDiagType_t*/, n /*int64_t*/,
// cusolverDnXtrtri-NEXT:                    a_type /*cudaDataType*/, a /*void **/, lda /*int64_t*/,
// cusolverDnXtrtri-NEXT:                    device_buffer /*void **/, device_buffer_size /*size_t*/,
// cusolverDnXtrtri-NEXT:                    host_buffer /*void **/, host_buffer_size /*size_t*/,
// cusolverDnXtrtri-NEXT:                    info /*int **/);
// cusolverDnXtrtri-NEXT: Is migrated to:
// cusolverDnXtrtri-NEXT:   dpct::lapack::trtri(*handle, uplo, diag, n, a_type, a, lda, device_buffer, device_buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXtrtri_bufferSize | FileCheck %s -check-prefix=cusolverDnXtrtri_bufferSize
// cusolverDnXtrtri_bufferSize: CUDA API:
// cusolverDnXtrtri_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXtrtri_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXtrtri_bufferSize-NEXT:   cusolverDnXtrtri_bufferSize(
// cusolverDnXtrtri_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnXtrtri_bufferSize-NEXT:       diag /*cublasDiagType_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
// cusolverDnXtrtri_bufferSize-NEXT:       a /*void **/, lda /*int64_t*/, &device_buffer_size /*size_t **/,
// cusolverDnXtrtri_bufferSize-NEXT:       &host_buffer_size /*size_t **/);
// cusolverDnXtrtri_bufferSize-NEXT: Is migrated to:
// cusolverDnXtrtri_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXtrtri_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXtrtri_bufferSize-NEXT:   dpct::lapack::trtri_scratchpad_size(*handle, uplo, diag, n, a_type, lda, &device_buffer_size, &host_buffer_size);
