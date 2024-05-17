// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1
// UNSUPPORTED: v8.0, v9.0, v9.1

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCunmtr | FileCheck %s -check-prefix=cusolverDnCunmtr
// cusolverDnCunmtr: CUDA API:
// cusolverDnCunmtr-NEXT:   cusolverDnCunmtr(handle /*cusolverDnHandle_t*/,
// cusolverDnCunmtr-NEXT:                    left_right /*cublasSideMode_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnCunmtr-NEXT:                    trans /*cublasOperation_t*/, m /*int*/, n /*int*/,
// cusolverDnCunmtr-NEXT:                    a /*cuComplex **/, lda /*int*/, tau /*cuComplex **/,
// cusolverDnCunmtr-NEXT:                    c /*cuComplex **/, ldc /*int*/, buffer /*cuComplex **/,
// cusolverDnCunmtr-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnCunmtr-NEXT: Is migrated to:
// cusolverDnCunmtr-NEXT:   oneapi::mkl::lapack::unmtr(*handle /*cusolverDnHandle_t*/,
// cusolverDnCunmtr-NEXT:                    left_right /*cublasSideMode_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnCunmtr-NEXT:                    trans /*cublasOperation_t*/, m /*int*/, n /*int*/,
// cusolverDnCunmtr-NEXT:                    (std::complex<float>*)a /*cuComplex **/, lda /*int*/, (std::complex<float>*)tau /*cuComplex **/,
// cusolverDnCunmtr-NEXT:                    (std::complex<float>*)c /*cuComplex **/, ldc /*int*/, (std::complex<float>*)buffer /*cuComplex **/,
// cusolverDnCunmtr-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZunmtr | FileCheck %s -check-prefix=cusolverDnZunmtr
// cusolverDnZunmtr: CUDA API:
// cusolverDnZunmtr-NEXT:   cusolverDnZunmtr(
// cusolverDnZunmtr-NEXT:       handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnZunmtr-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnZunmtr-NEXT:       n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZunmtr-NEXT:       tau /*cuDoubleComplex **/, c /*cuDoubleComplex **/, ldc /*int*/,
// cusolverDnZunmtr-NEXT:       buffer /*cuDoubleComplex **/, buffer_size /*int*/, info /*int **/);
// cusolverDnZunmtr-NEXT: Is migrated to:
// cusolverDnZunmtr-NEXT:   oneapi::mkl::lapack::unmtr(
// cusolverDnZunmtr-NEXT:       *handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
// cusolverDnZunmtr-NEXT:       uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cusolverDnZunmtr-NEXT:       n /*int*/, (std::complex<double>*)a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZunmtr-NEXT:       (std::complex<double>*)tau /*cuDoubleComplex **/, (std::complex<double>*)c /*cuDoubleComplex **/, ldc /*int*/,
// cusolverDnZunmtr-NEXT:       (std::complex<double>*)buffer /*cuDoubleComplex **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSorgtr_bufferSize | FileCheck %s -check-prefix=cusolverDnSorgtr_bufferSize
// cusolverDnSorgtr_bufferSize: CUDA API:
// cusolverDnSorgtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnSorgtr_bufferSize-NEXT:   cusolverDnSorgtr_bufferSize(handle /*cusolverDnHandle_t*/,
// cusolverDnSorgtr_bufferSize-NEXT:                               uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnSorgtr_bufferSize-NEXT:                               a /*const float **/, lda /*int*/,
// cusolverDnSorgtr_bufferSize-NEXT:                               tau /*const float **/, &buffer_size /*int **/);
// cusolverDnSorgtr_bufferSize-NEXT: Is migrated to:
// cusolverDnSorgtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnSorgtr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::orgtr_scratchpad_size<float>(*handle /*cusolverDnHandle_t*/,
// cusolverDnSorgtr_bufferSize-NEXT:                               uplo /*cublasFillMode_t*/, n /*const float **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDorgtr_bufferSize | FileCheck %s -check-prefix=cusolverDnDorgtr_bufferSize
// cusolverDnDorgtr_bufferSize: CUDA API:
// cusolverDnDorgtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnDorgtr_bufferSize-NEXT:   cusolverDnDorgtr_bufferSize(handle /*cusolverDnHandle_t*/,
// cusolverDnDorgtr_bufferSize-NEXT:                               uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnDorgtr_bufferSize-NEXT:                               a /*const double **/, lda /*int*/,
// cusolverDnDorgtr_bufferSize-NEXT:                               tau /*const double **/, &buffer_size /*int **/);
// cusolverDnDorgtr_bufferSize-NEXT: Is migrated to:
// cusolverDnDorgtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnDorgtr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::orgtr_scratchpad_size<double>(*handle /*cusolverDnHandle_t*/,
// cusolverDnDorgtr_bufferSize-NEXT:                               uplo /*cublasFillMode_t*/, n /*const double **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCungtr_bufferSize | FileCheck %s -check-prefix=cusolverDnCungtr_bufferSize
// cusolverDnCungtr_bufferSize: CUDA API:
// cusolverDnCungtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnCungtr_bufferSize-NEXT:   cusolverDnCungtr_bufferSize(
// cusolverDnCungtr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnCungtr_bufferSize-NEXT:       a /*const cuComplex **/, lda /*int*/, tau /*const cuComplex **/,
// cusolverDnCungtr_bufferSize-NEXT:       &buffer_size /*int **/);
// cusolverDnCungtr_bufferSize-NEXT: Is migrated to:
// cusolverDnCungtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnCungtr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<float>>(
// cusolverDnCungtr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*const cuComplex **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZungtr_bufferSize | FileCheck %s -check-prefix=cusolverDnZungtr_bufferSize
// cusolverDnZungtr_bufferSize: CUDA API:
// cusolverDnZungtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnZungtr_bufferSize-NEXT:   cusolverDnZungtr_bufferSize(
// cusolverDnZungtr_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZungtr_bufferSize-NEXT:       a /*const cuDoubleComplex **/, lda /*int*/,
// cusolverDnZungtr_bufferSize-NEXT:       tau /*const cuDoubleComplex **/, &buffer_size /*int **/);
// cusolverDnZungtr_bufferSize-NEXT: Is migrated to:
// cusolverDnZungtr_bufferSize-NEXT:   int buffer_size;
// cusolverDnZungtr_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<double>>(
// cusolverDnZungtr_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*const cuDoubleComplex **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSorgtr | FileCheck %s -check-prefix=cusolverDnSorgtr
// cusolverDnSorgtr: CUDA API:
// cusolverDnSorgtr-NEXT:   cusolverDnSorgtr(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSorgtr-NEXT:                    n /*int*/, a /*float **/, lda /*int*/, tau /*const float **/,
// cusolverDnSorgtr-NEXT:                    buffer /*float **/, buffer_size /*int*/, info /*int **/);
// cusolverDnSorgtr-NEXT: Is migrated to:
// cusolverDnSorgtr-NEXT:   oneapi::mkl::lapack::orgtr(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSorgtr-NEXT:                    n /*int*/, (float*)a /*float **/, lda /*int*/, (float*)tau /*const float **/,
// cusolverDnSorgtr-NEXT:                    (float*)buffer /*float **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDorgtr | FileCheck %s -check-prefix=cusolverDnDorgtr
// cusolverDnDorgtr: CUDA API:
// cusolverDnDorgtr-NEXT:   cusolverDnDorgtr(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDorgtr-NEXT:                    n /*int*/, a /*double **/, lda /*int*/,
// cusolverDnDorgtr-NEXT:                    tau /*const double **/, buffer /*double **/,
// cusolverDnDorgtr-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnDorgtr-NEXT: Is migrated to:
// cusolverDnDorgtr-NEXT:   oneapi::mkl::lapack::orgtr(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDorgtr-NEXT:                    n /*int*/, (double*)a /*double **/, lda /*int*/,
// cusolverDnDorgtr-NEXT:                    (double*)tau /*const double **/, (double*)buffer /*double **/,
// cusolverDnDorgtr-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCungtr | FileCheck %s -check-prefix=cusolverDnCungtr
// cusolverDnCungtr: CUDA API:
// cusolverDnCungtr-NEXT:   cusolverDnCungtr(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnCungtr-NEXT:                    n /*int*/, a /*cuComplex **/, lda /*int*/,
// cusolverDnCungtr-NEXT:                    tau /*const cuComplex **/, buffer /*cuComplex **/,
// cusolverDnCungtr-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnCungtr-NEXT: Is migrated to:
// cusolverDnCungtr-NEXT:   oneapi::mkl::lapack::ungtr(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnCungtr-NEXT:                    n /*int*/, (std::complex<float>*)a /*cuComplex **/, lda /*int*/,
// cusolverDnCungtr-NEXT:                    (std::complex<float>*)tau /*const cuComplex **/, (std::complex<float>*)buffer /*cuComplex **/,
// cusolverDnCungtr-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZungtr | FileCheck %s -check-prefix=cusolverDnZungtr
// cusolverDnZungtr: CUDA API:
// cusolverDnZungtr-NEXT:   cusolverDnZungtr(
// cusolverDnZungtr-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZungtr-NEXT:       a /*cuDoubleComplex **/, lda /*int*/, tau /*const cuDoubleComplex **/,
// cusolverDnZungtr-NEXT:       buffer /*cuDoubleComplex **/, buffer_size /*int*/, info /*int **/);
// cusolverDnZungtr-NEXT: Is migrated to:
// cusolverDnZungtr-NEXT:   oneapi::mkl::lapack::ungtr(
// cusolverDnZungtr-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZungtr-NEXT:       (std::complex<double>*)a /*cuDoubleComplex **/, lda /*int*/, (std::complex<double>*)tau /*const cuDoubleComplex **/,
// cusolverDnZungtr-NEXT:       (std::complex<double>*)buffer /*cuDoubleComplex **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSgesvd_bufferSize | FileCheck %s -check-prefix=cusolverDnSgesvd_bufferSize
// cusolverDnSgesvd_bufferSize: CUDA API:
// cusolverDnSgesvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnSgesvd_bufferSize-NEXT:   cusolverDnSgesvd_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnSgesvd_bufferSize-NEXT:                               n /*int*/, &buffer_size /*int **/);
// cusolverDnSgesvd_bufferSize-NEXT: Is migrated to:
// cusolverDnSgesvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnSgesvd_bufferSize-NEXT:   {
// cusolverDnSgesvd_bufferSize-NEXT:   oneapi::mkl::jobsvd job_ct_mkl_jobu;
// cusolverDnSgesvd_bufferSize-NEXT:   oneapi::mkl::jobsvd job_ct_mkl_jobvt;
// cusolverDnSgesvd_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::gesvd_scratchpad_size<float>(*handle, job_ct_mkl_jobu, job_ct_mkl_jobvt /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnSgesvd_bufferSize-NEXT:                               n, m, m, n /*int **/);
// cusolverDnSgesvd_bufferSize-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDgesvd_bufferSize | FileCheck %s -check-prefix=cusolverDnDgesvd_bufferSize
// cusolverDnDgesvd_bufferSize: CUDA API:
// cusolverDnDgesvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnDgesvd_bufferSize-NEXT:   cusolverDnDgesvd_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnDgesvd_bufferSize-NEXT:                               n /*int*/, &buffer_size /*int **/);
// cusolverDnDgesvd_bufferSize-NEXT: Is migrated to:
// cusolverDnDgesvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnDgesvd_bufferSize-NEXT:   {
// cusolverDnDgesvd_bufferSize-NEXT:   oneapi::mkl::jobsvd job_ct_mkl_jobu;
// cusolverDnDgesvd_bufferSize-NEXT:   oneapi::mkl::jobsvd job_ct_mkl_jobvt;
// cusolverDnDgesvd_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(*handle, job_ct_mkl_jobu, job_ct_mkl_jobvt /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnDgesvd_bufferSize-NEXT:                               n, m, m, n /*int **/);
// cusolverDnDgesvd_bufferSize-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCgesvd_bufferSize | FileCheck %s -check-prefix=cusolverDnCgesvd_bufferSize
// cusolverDnCgesvd_bufferSize: CUDA API:
// cusolverDnCgesvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnCgesvd_bufferSize-NEXT:   cusolverDnCgesvd_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnCgesvd_bufferSize-NEXT:                               n /*int*/, &buffer_size /*int **/);
// cusolverDnCgesvd_bufferSize-NEXT: Is migrated to:
// cusolverDnCgesvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnCgesvd_bufferSize-NEXT:   {
// cusolverDnCgesvd_bufferSize-NEXT:   oneapi::mkl::jobsvd job_ct_mkl_jobu;
// cusolverDnCgesvd_bufferSize-NEXT:   oneapi::mkl::jobsvd job_ct_mkl_jobvt;
// cusolverDnCgesvd_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<float>>(*handle, job_ct_mkl_jobu, job_ct_mkl_jobvt /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnCgesvd_bufferSize-NEXT:                               n, m, m, n /*int **/);
// cusolverDnCgesvd_bufferSize-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZgesvd_bufferSize | FileCheck %s -check-prefix=cusolverDnZgesvd_bufferSize
// cusolverDnZgesvd_bufferSize: CUDA API:
// cusolverDnZgesvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnZgesvd_bufferSize-NEXT:   cusolverDnZgesvd_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnZgesvd_bufferSize-NEXT:                               n /*int*/, &buffer_size /*int **/);
// cusolverDnZgesvd_bufferSize-NEXT: Is migrated to:
// cusolverDnZgesvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnZgesvd_bufferSize-NEXT:   {
// cusolverDnZgesvd_bufferSize-NEXT:   oneapi::mkl::jobsvd job_ct_mkl_jobu;
// cusolverDnZgesvd_bufferSize-NEXT:   oneapi::mkl::jobsvd job_ct_mkl_jobvt;
// cusolverDnZgesvd_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<double>>(*handle, job_ct_mkl_jobu, job_ct_mkl_jobvt /*cusolverDnHandle_t*/, m /*int*/,
// cusolverDnZgesvd_bufferSize-NEXT:                               n, m, m, n /*int **/);
// cusolverDnZgesvd_bufferSize-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSgesvd | FileCheck %s -check-prefix=cusolverDnSgesvd
// cusolverDnSgesvd: CUDA API:
// cusolverDnSgesvd-NEXT:   cusolverDnSgesvd(handle /*cusolverDnHandle_t*/, jobu /*signed char*/,
// cusolverDnSgesvd-NEXT:                    jobvt /*signed char*/, m /*int*/, n /*int*/, a /*float **/,
// cusolverDnSgesvd-NEXT:                    lda /*int*/, s /*float **/, u /*float **/, ldu /*int*/,
// cusolverDnSgesvd-NEXT:                    vt /*float **/, ldvt /*int*/, buffer /*float **/,
// cusolverDnSgesvd-NEXT:                    buffer_size /*int*/, buffer_for_real /*float **/,
// cusolverDnSgesvd-NEXT:                    info /*int **/);
// cusolverDnSgesvd-NEXT: Is migrated to:
// cusolverDnSgesvd-NEXT:   oneapi::mkl::lapack::gesvd(*handle /*cusolverDnHandle_t*/, (oneapi::mkl::jobsvd)jobu /*signed char*/,
// cusolverDnSgesvd-NEXT:                    (oneapi::mkl::jobsvd)jobvt /*signed char*/, m /*int*/, n /*int*/, (float*)a /*float **/,
// cusolverDnSgesvd-NEXT:                    lda /*int*/, (float*)s /*float **/, (float*)u /*float **/, ldu /*int*/,
// cusolverDnSgesvd-NEXT:                    (float*)vt /*float **/, ldvt /*int*/, (float*)buffer /*float **/,
// cusolverDnSgesvd-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDgesvd | FileCheck %s -check-prefix=cusolverDnDgesvd
// cusolverDnDgesvd: CUDA API:
// cusolverDnDgesvd-NEXT:   cusolverDnDgesvd(handle /*cusolverDnHandle_t*/, jobu /*signed char*/,
// cusolverDnDgesvd-NEXT:                    jobvt /*signed char*/, m /*int*/, n /*int*/, a /*double **/,
// cusolverDnDgesvd-NEXT:                    lda /*int*/, s /*double **/, u /*double **/, ldu /*int*/,
// cusolverDnDgesvd-NEXT:                    vt /*double **/, ldvt /*int*/, buffer /*double **/,
// cusolverDnDgesvd-NEXT:                    buffer_size /*int*/, buffer_for_real /*double **/,
// cusolverDnDgesvd-NEXT:                    info /*int **/);
// cusolverDnDgesvd-NEXT: Is migrated to:
// cusolverDnDgesvd-NEXT:   oneapi::mkl::lapack::gesvd(*handle /*cusolverDnHandle_t*/, (oneapi::mkl::jobsvd)jobu /*signed char*/,
// cusolverDnDgesvd-NEXT:                    (oneapi::mkl::jobsvd)jobvt /*signed char*/, m /*int*/, n /*int*/, (double*)a /*double **/,
// cusolverDnDgesvd-NEXT:                    lda /*int*/, (double*)s /*double **/, (double*)u /*double **/, ldu /*int*/,
// cusolverDnDgesvd-NEXT:                    (double*)vt /*double **/, ldvt /*int*/, (double*)buffer /*double **/,
// cusolverDnDgesvd-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCgesvd | FileCheck %s -check-prefix=cusolverDnCgesvd
// cusolverDnCgesvd: CUDA API:
// cusolverDnCgesvd-NEXT:   cusolverDnCgesvd(handle /*cusolverDnHandle_t*/, jobu /*signed char*/,
// cusolverDnCgesvd-NEXT:                    jobvt /*signed char*/, m /*int*/, n /*int*/,
// cusolverDnCgesvd-NEXT:                    a /*cuComplex **/, lda /*int*/, s /*float **/,
// cusolverDnCgesvd-NEXT:                    u /*cuComplex **/, ldu /*int*/, vt /*cuComplex **/,
// cusolverDnCgesvd-NEXT:                    ldvt /*int*/, buffer /*cuComplex **/, buffer_size /*int*/,
// cusolverDnCgesvd-NEXT:                    buffer_for_real /*float **/, info /*int **/);
// cusolverDnCgesvd-NEXT: Is migrated to:
// cusolverDnCgesvd-NEXT:   oneapi::mkl::lapack::gesvd(*handle /*cusolverDnHandle_t*/, (oneapi::mkl::jobsvd)jobu /*signed char*/,
// cusolverDnCgesvd-NEXT:                    (oneapi::mkl::jobsvd)jobvt /*signed char*/, m /*int*/, n /*int*/,
// cusolverDnCgesvd-NEXT:                    (std::complex<float>*)a /*cuComplex **/, lda /*int*/, (float*)s /*float **/,
// cusolverDnCgesvd-NEXT:                    (std::complex<float>*)u /*cuComplex **/, ldu /*int*/, (std::complex<float>*)vt /*cuComplex **/,
// cusolverDnCgesvd-NEXT:                    ldvt /*int*/, (std::complex<float>*)buffer /*cuComplex **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZgesvd | FileCheck %s -check-prefix=cusolverDnZgesvd
// cusolverDnZgesvd: CUDA API:
// cusolverDnZgesvd-NEXT:   cusolverDnZgesvd(
// cusolverDnZgesvd-NEXT:       handle /*cusolverDnHandle_t*/, jobu /*signed char*/,
// cusolverDnZgesvd-NEXT:       jobvt /*signed char*/, m /*int*/, n /*int*/, a /*cuDoubleComplex **/,
// cusolverDnZgesvd-NEXT:       lda /*int*/, s /*double **/, u /*cuDoubleComplex **/, ldu /*int*/,
// cusolverDnZgesvd-NEXT:       vt /*cuDoubleComplex **/, ldvt /*int*/, buffer /*cuDoubleComplex **/,
// cusolverDnZgesvd-NEXT:       buffer_size /*int*/, buffer_for_real /*double **/, info /*int **/);
// cusolverDnZgesvd-NEXT: Is migrated to:
// cusolverDnZgesvd-NEXT:   oneapi::mkl::lapack::gesvd(
// cusolverDnZgesvd-NEXT:       *handle /*cusolverDnHandle_t*/, (oneapi::mkl::jobsvd)jobu /*signed char*/,
// cusolverDnZgesvd-NEXT:       (oneapi::mkl::jobsvd)jobvt /*signed char*/, m /*int*/, n /*int*/, (std::complex<double>*)a /*cuDoubleComplex **/,
// cusolverDnZgesvd-NEXT:       lda /*int*/, (double*)s /*double **/, (std::complex<double>*)u /*cuDoubleComplex **/, ldu /*int*/,
// cusolverDnZgesvd-NEXT:       (std::complex<double>*)vt /*cuDoubleComplex **/, ldvt /*int*/, (std::complex<double>*)buffer /*cuDoubleComplex **/,
// cusolverDnZgesvd-NEXT:       buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSgesvdj_bufferSize | FileCheck %s -check-prefix=cusolverDnSgesvdj_bufferSize
// cusolverDnSgesvdj_bufferSize: CUDA API:
// cusolverDnSgesvdj_bufferSize-NEXT:   int buffer_size;
// cusolverDnSgesvdj_bufferSize-NEXT:   cusolverDnSgesvdj_bufferSize(
// cusolverDnSgesvdj_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/, econ /*int*/,
// cusolverDnSgesvdj_bufferSize-NEXT:       m /*int*/, n /*int*/, a /*const float **/, lda /*int*/,
// cusolverDnSgesvdj_bufferSize-NEXT:       s /*const float **/, u /*const float **/, ldu /*int*/,
// cusolverDnSgesvdj_bufferSize-NEXT:       v /*const float **/, ldv /*int*/, &buffer_size /*int **/,
// cusolverDnSgesvdj_bufferSize-NEXT:       params /*gesvdjInfo_t*/);
// cusolverDnSgesvdj_bufferSize-NEXT: Is migrated to:
// cusolverDnSgesvdj_bufferSize-NEXT:   int buffer_size;
// cusolverDnSgesvdj_bufferSize-NEXT:   dpct::lapack::gesvd_scratchpad_size(*handle, jobz, econ, m, n, dpct::library_data_t::real_float, lda, dpct::library_data_t::real_float, ldu, dpct::library_data_t::real_float, ldv, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDgesvdj_bufferSize | FileCheck %s -check-prefix=cusolverDnDgesvdj_bufferSize
// cusolverDnDgesvdj_bufferSize: CUDA API:
// cusolverDnDgesvdj_bufferSize-NEXT:   int buffer_size;
// cusolverDnDgesvdj_bufferSize-NEXT:   cusolverDnDgesvdj_bufferSize(
// cusolverDnDgesvdj_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/, econ /*int*/,
// cusolverDnDgesvdj_bufferSize-NEXT:       m /*int*/, n /*int*/, a /*const double **/, lda /*int*/,
// cusolverDnDgesvdj_bufferSize-NEXT:       s /*const double **/, u /*const double **/, ldu /*int*/,
// cusolverDnDgesvdj_bufferSize-NEXT:       v /*const double **/, ldv /*int*/, &buffer_size /*int **/,
// cusolverDnDgesvdj_bufferSize-NEXT:       params /*gesvdjInfo_t*/);
// cusolverDnDgesvdj_bufferSize-NEXT: Is migrated to:
// cusolverDnDgesvdj_bufferSize-NEXT:   int buffer_size;
// cusolverDnDgesvdj_bufferSize-NEXT:   dpct::lapack::gesvd_scratchpad_size(*handle, jobz, econ, m, n, dpct::library_data_t::real_double, lda, dpct::library_data_t::real_double, ldu, dpct::library_data_t::real_double, ldv, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCgesvdj_bufferSize | FileCheck %s -check-prefix=cusolverDnCgesvdj_bufferSize
// cusolverDnCgesvdj_bufferSize: CUDA API:
// cusolverDnCgesvdj_bufferSize-NEXT:   int buffer_size;
// cusolverDnCgesvdj_bufferSize-NEXT:   cusolverDnCgesvdj_bufferSize(
// cusolverDnCgesvdj_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/, econ /*int*/,
// cusolverDnCgesvdj_bufferSize-NEXT:       m /*int*/, n /*int*/, a /*const cuComplex **/, lda /*int*/,
// cusolverDnCgesvdj_bufferSize-NEXT:       s /*const float **/, u /*const cuComplex **/, ldu /*int*/,
// cusolverDnCgesvdj_bufferSize-NEXT:       v /*const cuComplex **/, ldv /*int*/, &buffer_size /*int **/,
// cusolverDnCgesvdj_bufferSize-NEXT:       params /*gesvdjInfo_t*/);
// cusolverDnCgesvdj_bufferSize-NEXT: Is migrated to:
// cusolverDnCgesvdj_bufferSize-NEXT:   int buffer_size;
// cusolverDnCgesvdj_bufferSize-NEXT:   dpct::lapack::gesvd_scratchpad_size(*handle, jobz, econ, m, n, dpct::library_data_t::complex_float, lda, dpct::library_data_t::complex_float, ldu, dpct::library_data_t::complex_float, ldv, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZgesvdj_bufferSize | FileCheck %s -check-prefix=cusolverDnZgesvdj_bufferSize
// cusolverDnZgesvdj_bufferSize: CUDA API:
// cusolverDnZgesvdj_bufferSize-NEXT:   int buffer_size;
// cusolverDnZgesvdj_bufferSize-NEXT:   cusolverDnZgesvdj_bufferSize(
// cusolverDnZgesvdj_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/, econ /*int*/,
// cusolverDnZgesvdj_bufferSize-NEXT:       m /*int*/, n /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
// cusolverDnZgesvdj_bufferSize-NEXT:       s /*const double **/, u /*const cuDoubleComplex **/, ldu /*int*/,
// cusolverDnZgesvdj_bufferSize-NEXT:       v /*const cuDoubleComplex **/, ldv /*int*/, &buffer_size /*int **/,
// cusolverDnZgesvdj_bufferSize-NEXT:       params /*gesvdjInfo_t*/);
// cusolverDnZgesvdj_bufferSize-NEXT: Is migrated to:
// cusolverDnZgesvdj_bufferSize-NEXT:   int buffer_size;
// cusolverDnZgesvdj_bufferSize-NEXT:   dpct::lapack::gesvd_scratchpad_size(*handle, jobz, econ, m, n, dpct::library_data_t::complex_double, lda, dpct::library_data_t::complex_double, ldu, dpct::library_data_t::complex_double, ldv, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSgesvdj | FileCheck %s -check-prefix=cusolverDnSgesvdj
// cusolverDnSgesvdj: CUDA API:
// cusolverDnSgesvdj-NEXT:   cusolverDnSgesvdj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnSgesvdj-NEXT:                     econ /*int*/, m /*int*/, n /*int*/, a /*float **/,
// cusolverDnSgesvdj-NEXT:                     lda /*int*/, s /*float **/, u /*float **/, ldu /*int*/,
// cusolverDnSgesvdj-NEXT:                     v /*float **/, ldv /*int*/, buffer /*float **/,
// cusolverDnSgesvdj-NEXT:                     buffer_size /*int*/, info /*int **/,
// cusolverDnSgesvdj-NEXT:                     params /*gesvdjInfo_t*/);
// cusolverDnSgesvdj-NEXT: Is migrated to:
// cusolverDnSgesvdj-NEXT:   dpct::lapack::gesvd(*handle, jobz, econ, m, n, dpct::library_data_t::real_float, a, lda, dpct::library_data_t::real_float, s, dpct::library_data_t::real_float, u, ldu, dpct::library_data_t::real_float, v, ldv, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDgesvdj | FileCheck %s -check-prefix=cusolverDnDgesvdj
// cusolverDnDgesvdj: CUDA API:
// cusolverDnDgesvdj-NEXT:   cusolverDnDgesvdj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnDgesvdj-NEXT:                     econ /*int*/, m /*int*/, n /*int*/, a /*double **/,
// cusolverDnDgesvdj-NEXT:                     lda /*int*/, s /*double **/, u /*double **/, ldu /*int*/,
// cusolverDnDgesvdj-NEXT:                     v /*double **/, ldv /*int*/, buffer /*double **/,
// cusolverDnDgesvdj-NEXT:                     buffer_size /*int*/, info /*int **/,
// cusolverDnDgesvdj-NEXT:                     params /*gesvdjInfo_t*/);
// cusolverDnDgesvdj-NEXT: Is migrated to:
// cusolverDnDgesvdj-NEXT:   dpct::lapack::gesvd(*handle, jobz, econ, m, n, dpct::library_data_t::real_double, a, lda, dpct::library_data_t::real_double, s, dpct::library_data_t::real_double, u, ldu, dpct::library_data_t::real_double, v, ldv, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCgesvdj | FileCheck %s -check-prefix=cusolverDnCgesvdj
// cusolverDnCgesvdj: CUDA API:
// cusolverDnCgesvdj-NEXT:   cusolverDnCgesvdj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnCgesvdj-NEXT:                     econ /*int*/, m /*int*/, n /*int*/, a /*cuComplex **/,
// cusolverDnCgesvdj-NEXT:                     lda /*int*/, s /*float **/, u /*cuComplex **/, ldu /*int*/,
// cusolverDnCgesvdj-NEXT:                     v /*cuComplex **/, ldv /*int*/, buffer /*cuComplex **/,
// cusolverDnCgesvdj-NEXT:                     buffer_size /*int*/, info /*int **/,
// cusolverDnCgesvdj-NEXT:                     params /*gesvdjInfo_t*/);
// cusolverDnCgesvdj-NEXT: Is migrated to:
// cusolverDnCgesvdj-NEXT:   dpct::lapack::gesvd(*handle, jobz, econ, m, n, dpct::library_data_t::complex_float, a, lda, dpct::library_data_t::real_float, s, dpct::library_data_t::complex_float, u, ldu, dpct::library_data_t::complex_float, v, ldv, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZgesvdj | FileCheck %s -check-prefix=cusolverDnZgesvdj
// cusolverDnZgesvdj: CUDA API:
// cusolverDnZgesvdj-NEXT:   cusolverDnZgesvdj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnZgesvdj-NEXT:                     econ /*int*/, m /*int*/, n /*int*/, a /*cuDoubleComplex **/,
// cusolverDnZgesvdj-NEXT:                     lda /*int*/, s /*double **/, u /*cuDoubleComplex **/,
// cusolverDnZgesvdj-NEXT:                     ldu /*int*/, v /*cuDoubleComplex **/, ldv /*int*/,
// cusolverDnZgesvdj-NEXT:                     buffer /*cuDoubleComplex **/, buffer_size /*int*/,
// cusolverDnZgesvdj-NEXT:                     info /*int **/, params /*gesvdjInfo_t*/);
// cusolverDnZgesvdj-NEXT: Is migrated to:
// cusolverDnZgesvdj-NEXT:   dpct::lapack::gesvd(*handle, jobz, econ, m, n, dpct::library_data_t::complex_double, a, lda, dpct::library_data_t::real_double, s, dpct::library_data_t::complex_double, u, ldu, dpct::library_data_t::complex_double, v, ldv, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsyevd_bufferSize | FileCheck %s -check-prefix=cusolverDnSsyevd_bufferSize
// cusolverDnSsyevd_bufferSize: CUDA API:
// cusolverDnSsyevd_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsyevd_bufferSize-NEXT:   cusolverDnSsyevd_bufferSize(
// cusolverDnSsyevd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnSsyevd_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int*/, a /*const float **/, lda /*int*/,
// cusolverDnSsyevd_bufferSize-NEXT:       w /*const float **/, &buffer_size /*int **/);
// cusolverDnSsyevd_bufferSize-NEXT: Is migrated to:
// cusolverDnSsyevd_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsyevd_bufferSize-NEXT:   dpct::lapack::syheevd_scratchpad_size<float>(*handle, jobz, uplo, n, lda, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsyevd_bufferSize | FileCheck %s -check-prefix=cusolverDnDsyevd_bufferSize
// cusolverDnDsyevd_bufferSize: CUDA API:
// cusolverDnDsyevd_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsyevd_bufferSize-NEXT:   cusolverDnDsyevd_bufferSize(
// cusolverDnDsyevd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnDsyevd_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int*/, a /*const double **/, lda /*int*/,
// cusolverDnDsyevd_bufferSize-NEXT:       w /*const double **/, &buffer_size /*int **/);
// cusolverDnDsyevd_bufferSize-NEXT: Is migrated to:
// cusolverDnDsyevd_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsyevd_bufferSize-NEXT:   dpct::lapack::syheevd_scratchpad_size<double>(*handle, jobz, uplo, n, lda, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCheevd_bufferSize | FileCheck %s -check-prefix=cusolverDnCheevd_bufferSize
// cusolverDnCheevd_bufferSize: CUDA API:
// cusolverDnCheevd_bufferSize-NEXT:   int buffer_size;
// cusolverDnCheevd_bufferSize-NEXT:   cusolverDnCheevd_bufferSize(
// cusolverDnCheevd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnCheevd_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int*/, a /*const cuComplex **/,
// cusolverDnCheevd_bufferSize-NEXT:       lda /*int*/, w /*const float **/, &buffer_size /*int **/);
// cusolverDnCheevd_bufferSize-NEXT: Is migrated to:
// cusolverDnCheevd_bufferSize-NEXT:   int buffer_size;
// cusolverDnCheevd_bufferSize-NEXT:   dpct::lapack::syheevd_scratchpad_size<std::complex<float>>(*handle, jobz, uplo, n, lda, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZheevd_bufferSize | FileCheck %s -check-prefix=cusolverDnZheevd_bufferSize
// cusolverDnZheevd_bufferSize: CUDA API:
// cusolverDnZheevd_bufferSize-NEXT:   int buffer_size;
// cusolverDnZheevd_bufferSize-NEXT:   cusolverDnZheevd_bufferSize(
// cusolverDnZheevd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnZheevd_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int*/, a /*const cuDoubleComplex **/,
// cusolverDnZheevd_bufferSize-NEXT:       lda /*int*/, w /*const double **/, &buffer_size /*int **/);
// cusolverDnZheevd_bufferSize-NEXT: Is migrated to:
// cusolverDnZheevd_bufferSize-NEXT:   int buffer_size;
// cusolverDnZheevd_bufferSize-NEXT:   dpct::lapack::syheevd_scratchpad_size<std::complex<double>>(*handle, jobz, uplo, n, lda, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsyevd | FileCheck %s -check-prefix=cusolverDnSsyevd
// cusolverDnSsyevd: CUDA API:
// cusolverDnSsyevd-NEXT:   cusolverDnSsyevd(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnSsyevd-NEXT:                    uplo /*cublasFillMode_t*/, n /*int*/, a /*float **/,
// cusolverDnSsyevd-NEXT:                    lda /*int*/, w /*float **/, buffer /*float **/,
// cusolverDnSsyevd-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnSsyevd-NEXT: Is migrated to:
// cusolverDnSsyevd-NEXT:   dpct::lapack::syheevd<float, float>(*handle, jobz, uplo, n, a, lda, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsyevd | FileCheck %s -check-prefix=cusolverDnDsyevd
// cusolverDnDsyevd: CUDA API:
// cusolverDnDsyevd-NEXT:   cusolverDnDsyevd(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnDsyevd-NEXT:                    uplo /*cublasFillMode_t*/, n /*int*/, a /*double **/,
// cusolverDnDsyevd-NEXT:                    lda /*int*/, w /*double **/, buffer /*double **/,
// cusolverDnDsyevd-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnDsyevd-NEXT: Is migrated to:
// cusolverDnDsyevd-NEXT:   dpct::lapack::syheevd<double, double>(*handle, jobz, uplo, n, a, lda, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCheevd | FileCheck %s -check-prefix=cusolverDnCheevd
// cusolverDnCheevd: CUDA API:
// cusolverDnCheevd-NEXT:   cusolverDnCheevd(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnCheevd-NEXT:                    uplo /*cublasFillMode_t*/, n /*int*/, a /*cuComplex **/,
// cusolverDnCheevd-NEXT:                    lda /*int*/, w /*float **/, buffer /*cuComplex **/,
// cusolverDnCheevd-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnCheevd-NEXT: Is migrated to:
// cusolverDnCheevd-NEXT:   dpct::lapack::syheevd<sycl::float2, float>(*handle, jobz, uplo, n, a, lda, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZheevd | FileCheck %s -check-prefix=cusolverDnZheevd
// cusolverDnZheevd: CUDA API:
// cusolverDnZheevd-NEXT:   cusolverDnZheevd(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnZheevd-NEXT:                    uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZheevd-NEXT:                    a /*cuDoubleComplex **/, lda /*int*/, w /*double **/,
// cusolverDnZheevd-NEXT:                    buffer /*cuDoubleComplex **/, buffer_size /*int*/,
// cusolverDnZheevd-NEXT:                    info /*int **/);
// cusolverDnZheevd-NEXT: Is migrated to:
// cusolverDnZheevd-NEXT:   dpct::lapack::syheevd<sycl::double2, double>(*handle, jobz, uplo, n, a, lda, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsygvd_bufferSize | FileCheck %s -check-prefix=cusolverDnSsygvd_bufferSize
// cusolverDnSsygvd_bufferSize: CUDA API:
// cusolverDnSsygvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsygvd_bufferSize-NEXT:   cusolverDnSsygvd_bufferSize(
// cusolverDnSsygvd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnSsygvd_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnSsygvd_bufferSize-NEXT:       a /*const float **/, lda /*int*/, b /*const float **/, ldb /*int*/,
// cusolverDnSsygvd_bufferSize-NEXT:       w /*const float **/, &buffer_size /*int **/);
// cusolverDnSsygvd_bufferSize-NEXT: Is migrated to:
// cusolverDnSsygvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsygvd_bufferSize-NEXT:   buffer_size = oneapi::mkl::lapack::sygvd_scratchpad_size<float>(*handle, itype, jobz, uplo, n, lda, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsygvd_bufferSize | FileCheck %s -check-prefix=cusolverDnDsygvd_bufferSize
// cusolverDnDsygvd_bufferSize: CUDA API:
// cusolverDnDsygvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsygvd_bufferSize-NEXT:   cusolverDnDsygvd_bufferSize(
// cusolverDnDsygvd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnDsygvd_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnDsygvd_bufferSize-NEXT:       a /*const double **/, lda /*int*/, b /*const double **/, ldb /*int*/,
// cusolverDnDsygvd_bufferSize-NEXT:       w /*const double **/, &buffer_size /*int **/);
// cusolverDnDsygvd_bufferSize-NEXT: Is migrated to:
// cusolverDnDsygvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsygvd_bufferSize-NEXT:   buffer_size = oneapi::mkl::lapack::sygvd_scratchpad_size<double>(*handle, itype, jobz, uplo, n, lda, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnChegvd_bufferSize | FileCheck %s -check-prefix=cusolverDnChegvd_bufferSize
// cusolverDnChegvd_bufferSize: CUDA API:
// cusolverDnChegvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnChegvd_bufferSize-NEXT:   cusolverDnChegvd_bufferSize(
// cusolverDnChegvd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnChegvd_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnChegvd_bufferSize-NEXT:       a /*const cuComplex **/, lda /*int*/, b /*const cuComplex **/,
// cusolverDnChegvd_bufferSize-NEXT:       ldb /*int*/, w /*const float **/, &buffer_size /*int **/);
// cusolverDnChegvd_bufferSize-NEXT: Is migrated to:
// cusolverDnChegvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnChegvd_bufferSize-NEXT:   buffer_size = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<float>>(*handle, itype, jobz, uplo, n, lda, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZhegvd_bufferSize | FileCheck %s -check-prefix=cusolverDnZhegvd_bufferSize
// cusolverDnZhegvd_bufferSize: CUDA API:
// cusolverDnZhegvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnZhegvd_bufferSize-NEXT:   cusolverDnZhegvd_bufferSize(
// cusolverDnZhegvd_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnZhegvd_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZhegvd_bufferSize-NEXT:       a /*const cuDoubleComplex **/, lda /*int*/, b /*const cuDoubleComplex **/,
// cusolverDnZhegvd_bufferSize-NEXT:       ldb /*int*/, w /*const double **/, &buffer_size /*int **/);
// cusolverDnZhegvd_bufferSize-NEXT: Is migrated to:
// cusolverDnZhegvd_bufferSize-NEXT:   int buffer_size;
// cusolverDnZhegvd_bufferSize-NEXT:   buffer_size = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<double>>(*handle, itype, jobz, uplo, n, lda, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsygvd | FileCheck %s -check-prefix=cusolverDnSsygvd
// cusolverDnSsygvd: CUDA API:
// cusolverDnSsygvd-NEXT:   cusolverDnSsygvd(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnSsygvd-NEXT:                    jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSsygvd-NEXT:                    n /*int*/, a /*float **/, lda /*int*/, b /*float **/,
// cusolverDnSsygvd-NEXT:                    ldb /*int*/, w /*float **/, buffer /*float **/,
// cusolverDnSsygvd-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnSsygvd-NEXT: Is migrated to:
// cusolverDnSsygvd-NEXT:   dpct::lapack::sygvd(*handle, itype, jobz, uplo, n, a, lda, b, ldb, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsygvd | FileCheck %s -check-prefix=cusolverDnDsygvd
// cusolverDnDsygvd: CUDA API:
// cusolverDnDsygvd-NEXT:   cusolverDnDsygvd(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnDsygvd-NEXT:                    jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDsygvd-NEXT:                    n /*int*/, a /*double **/, lda /*int*/, b /*double **/,
// cusolverDnDsygvd-NEXT:                    ldb /*int*/, w /*double **/, buffer /*double **/,
// cusolverDnDsygvd-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnDsygvd-NEXT: Is migrated to:
// cusolverDnDsygvd-NEXT:   dpct::lapack::sygvd(*handle, itype, jobz, uplo, n, a, lda, b, ldb, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnChegvd | FileCheck %s -check-prefix=cusolverDnChegvd
// cusolverDnChegvd: CUDA API:
// cusolverDnChegvd-NEXT:   cusolverDnChegvd(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnChegvd-NEXT:                    jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnChegvd-NEXT:                    n /*int*/, a /*cuComplex **/, lda /*int*/, b /*cuComplex **/,
// cusolverDnChegvd-NEXT:                    ldb /*int*/, w /*float **/, buffer /*cuComplex **/,
// cusolverDnChegvd-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnChegvd-NEXT: Is migrated to:
// cusolverDnChegvd-NEXT:   dpct::lapack::hegvd(*handle, itype, jobz, uplo, n, a, lda, b, ldb, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZhegvd | FileCheck %s -check-prefix=cusolverDnZhegvd
// cusolverDnZhegvd: CUDA API:
// cusolverDnZhegvd-NEXT:   cusolverDnZhegvd(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnZhegvd-NEXT:                    jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnZhegvd-NEXT:                    n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZhegvd-NEXT:                    b /*cuDoubleComplex **/, ldb /*int*/, w /*double **/,
// cusolverDnZhegvd-NEXT:                    buffer /*cuDoubleComplex **/, buffer_size /*int*/,
// cusolverDnZhegvd-NEXT:                    info /*int **/);
// cusolverDnZhegvd-NEXT: Is migrated to:
// cusolverDnZhegvd-NEXT:   dpct::lapack::hegvd(*handle, itype, jobz, uplo, n, a, lda, b, ldb, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsyevj_bufferSize | FileCheck %s -check-prefix=cusolverDnSsyevj_bufferSize
// cusolverDnSsyevj_bufferSize: CUDA API:
// cusolverDnSsyevj_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsyevj_bufferSize-NEXT:   cusolverDnSsyevj_bufferSize(
// cusolverDnSsyevj_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnSsyevj_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int*/, a /*const float **/, lda /*int*/,
// cusolverDnSsyevj_bufferSize-NEXT:       w /*const float **/, &buffer_size /*int **/, params /*syevjInfo_t*/);
// cusolverDnSsyevj_bufferSize-NEXT: Is migrated to:
// cusolverDnSsyevj_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsyevj_bufferSize-NEXT:   dpct::lapack::syheev_scratchpad_size<float>(*handle, jobz, uplo, n, lda, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsyevj_bufferSize | FileCheck %s -check-prefix=cusolverDnDsyevj_bufferSize
// cusolverDnDsyevj_bufferSize: CUDA API:
// cusolverDnDsyevj_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsyevj_bufferSize-NEXT:   cusolverDnDsyevj_bufferSize(
// cusolverDnDsyevj_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnDsyevj_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int*/, a /*const double **/, lda /*int*/,
// cusolverDnDsyevj_bufferSize-NEXT:       w /*const double **/, &buffer_size /*int **/, params /*syevjInfo_t*/);
// cusolverDnDsyevj_bufferSize-NEXT: Is migrated to:
// cusolverDnDsyevj_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsyevj_bufferSize-NEXT:   dpct::lapack::syheev_scratchpad_size<double>(*handle, jobz, uplo, n, lda, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCheevj_bufferSize | FileCheck %s -check-prefix=cusolverDnCheevj_bufferSize
// cusolverDnCheevj_bufferSize: CUDA API:
// cusolverDnCheevj_bufferSize-NEXT:   int buffer_size;
// cusolverDnCheevj_bufferSize-NEXT:   cusolverDnCheevj_bufferSize(
// cusolverDnCheevj_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnCheevj_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int*/, a /*const cuComplex **/,
// cusolverDnCheevj_bufferSize-NEXT:       lda /*int*/, w /*const float **/, &buffer_size /*int **/,
// cusolverDnCheevj_bufferSize-NEXT:       params /*syevjInfo_t*/);
// cusolverDnCheevj_bufferSize-NEXT: Is migrated to:
// cusolverDnCheevj_bufferSize-NEXT:   int buffer_size;
// cusolverDnCheevj_bufferSize-NEXT:   dpct::lapack::syheev_scratchpad_size<sycl::float2>(*handle, jobz, uplo, n, lda, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZheevj_bufferSize | FileCheck %s -check-prefix=cusolverDnZheevj_bufferSize
// cusolverDnZheevj_bufferSize: CUDA API:
// cusolverDnZheevj_bufferSize-NEXT:   int buffer_size;
// cusolverDnZheevj_bufferSize-NEXT:   cusolverDnZheevj_bufferSize(
// cusolverDnZheevj_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnZheevj_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int*/, a /*const cuDoubleComplex **/,
// cusolverDnZheevj_bufferSize-NEXT:       lda /*int*/, w /*const double **/, &buffer_size /*int **/,
// cusolverDnZheevj_bufferSize-NEXT:       params /*syevjInfo_t*/);
// cusolverDnZheevj_bufferSize-NEXT: Is migrated to:
// cusolverDnZheevj_bufferSize-NEXT:   int buffer_size;
// cusolverDnZheevj_bufferSize-NEXT:   dpct::lapack::syheev_scratchpad_size<sycl::double2>(*handle, jobz, uplo, n, lda, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsyevj | FileCheck %s -check-prefix=cusolverDnSsyevj
// cusolverDnSsyevj: CUDA API:
// cusolverDnSsyevj-NEXT:   cusolverDnSsyevj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnSsyevj-NEXT:                    uplo /*cublasFillMode_t*/, n /*int*/, a /*float **/,
// cusolverDnSsyevj-NEXT:                    lda /*int*/, w /*float **/, buffer /*float **/,
// cusolverDnSsyevj-NEXT:                    buffer_size /*int*/, info /*int **/, params /*syevjInfo_t*/);
// cusolverDnSsyevj-NEXT: Is migrated to:
// cusolverDnSsyevj-NEXT:   dpct::lapack::syheev<float, float>(*handle, jobz, uplo, n, a, lda, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsyevj | FileCheck %s -check-prefix=cusolverDnDsyevj
// cusolverDnDsyevj: CUDA API:
// cusolverDnDsyevj-NEXT:   cusolverDnDsyevj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnDsyevj-NEXT:                    uplo /*cublasFillMode_t*/, n /*int*/, a /*double **/,
// cusolverDnDsyevj-NEXT:                    lda /*int*/, w /*double **/, buffer /*double **/,
// cusolverDnDsyevj-NEXT:                    buffer_size /*int*/, info /*int **/, params /*syevjInfo_t*/);
// cusolverDnDsyevj-NEXT: Is migrated to:
// cusolverDnDsyevj-NEXT:   dpct::lapack::syheev<double, double>(*handle, jobz, uplo, n, a, lda, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCheevj | FileCheck %s -check-prefix=cusolverDnCheevj
// cusolverDnCheevj: CUDA API:
// cusolverDnCheevj-NEXT:   cusolverDnCheevj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnCheevj-NEXT:                    uplo /*cublasFillMode_t*/, n /*int*/, a /*cuComplex **/,
// cusolverDnCheevj-NEXT:                    lda /*int*/, w /*float **/, buffer /*cuComplex **/,
// cusolverDnCheevj-NEXT:                    buffer_size /*int*/, info /*int **/, params /*syevjInfo_t*/);
// cusolverDnCheevj-NEXT: Is migrated to:
// cusolverDnCheevj-NEXT:   dpct::lapack::syheev<sycl::float2, float>(*handle, jobz, uplo, n, a, lda, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZheevj | FileCheck %s -check-prefix=cusolverDnZheevj
// cusolverDnZheevj: CUDA API:
// cusolverDnZheevj-NEXT:   cusolverDnZheevj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnZheevj-NEXT:                    uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZheevj-NEXT:                    a /*cuDoubleComplex **/, lda /*int*/, w /*double **/,
// cusolverDnZheevj-NEXT:                    buffer /*cuDoubleComplex **/, buffer_size /*int*/,
// cusolverDnZheevj-NEXT:                    info /*int **/, params /*syevjInfo_t*/);
// cusolverDnZheevj-NEXT: Is migrated to:
// cusolverDnZheevj-NEXT:   dpct::lapack::syheev<sycl::double2, double>(*handle, jobz, uplo, n, a, lda, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsygvj_bufferSize | FileCheck %s -check-prefix=cusolverDnSsygvj_bufferSize
// cusolverDnSsygvj_bufferSize: CUDA API:
// cusolverDnSsygvj_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsygvj_bufferSize-NEXT:   cusolverDnSsygvj_bufferSize(
// cusolverDnSsygvj_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnSsygvj_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnSsygvj_bufferSize-NEXT:       a /*const float **/, lda /*int*/, b /*const float **/, ldb /*int*/,
// cusolverDnSsygvj_bufferSize-NEXT:       w /*const float **/, &buffer_size /*int **/, params /*syevjInfo_t*/);
// cusolverDnSsygvj_bufferSize-NEXT: Is migrated to:
// cusolverDnSsygvj_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsygvj_bufferSize-NEXT:   dpct::lapack::syhegvd_scratchpad_size<float>(*handle, itype, jobz, uplo, n, lda, ldb, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsygvj_bufferSize | FileCheck %s -check-prefix=cusolverDnDsygvj_bufferSize
// cusolverDnDsygvj_bufferSize: CUDA API:
// cusolverDnDsygvj_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsygvj_bufferSize-NEXT:   cusolverDnDsygvj_bufferSize(
// cusolverDnDsygvj_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnDsygvj_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnDsygvj_bufferSize-NEXT:       a /*const double **/, lda /*int*/, b /*const double **/, ldb /*int*/,
// cusolverDnDsygvj_bufferSize-NEXT:       w /*const double **/, &buffer_size /*int **/, params /*syevjInfo_t*/);
// cusolverDnDsygvj_bufferSize-NEXT: Is migrated to:
// cusolverDnDsygvj_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsygvj_bufferSize-NEXT:   dpct::lapack::syhegvd_scratchpad_size<float>(*handle, itype, jobz, uplo, n, lda, ldb, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnChegvj_bufferSize | FileCheck %s -check-prefix=cusolverDnChegvj_bufferSize
// cusolverDnChegvj_bufferSize: CUDA API:
// cusolverDnChegvj_bufferSize-NEXT:   int buffer_size;
// cusolverDnChegvj_bufferSize-NEXT:   cusolverDnChegvj_bufferSize(
// cusolverDnChegvj_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnChegvj_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnChegvj_bufferSize-NEXT:       a /*const cuComplex **/, lda /*int*/, b /*const cuComplex **/,
// cusolverDnChegvj_bufferSize-NEXT:       ldb /*int*/, w /*const float **/, &buffer_size /*int **/,
// cusolverDnChegvj_bufferSize-NEXT:       params /*syevjInfo_t*/);
// cusolverDnChegvj_bufferSize-NEXT: Is migrated to:
// cusolverDnChegvj_bufferSize-NEXT:   int buffer_size;
// cusolverDnChegvj_bufferSize-NEXT:   dpct::lapack::syhegvd_scratchpad_size<sycl::float2>(*handle, itype, jobz, uplo, n, lda, ldb, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZhegvj_bufferSize | FileCheck %s -check-prefix=cusolverDnZhegvj_bufferSize
// cusolverDnZhegvj_bufferSize: CUDA API:
// cusolverDnZhegvj_bufferSize-NEXT:   int buffer_size;
// cusolverDnZhegvj_bufferSize-NEXT:   cusolverDnZhegvj_bufferSize(
// cusolverDnZhegvj_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnZhegvj_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZhegvj_bufferSize-NEXT:       a /*const cuDoubleComplex **/, lda /*int*/, b /*const cuDoubleComplex **/,
// cusolverDnZhegvj_bufferSize-NEXT:       ldb /*int*/, w /*const double **/, &buffer_size /*int **/,
// cusolverDnZhegvj_bufferSize-NEXT:       params /*syevjInfo_t*/);
// cusolverDnZhegvj_bufferSize-NEXT: Is migrated to:
// cusolverDnZhegvj_bufferSize-NEXT:   int buffer_size;
// cusolverDnZhegvj_bufferSize-NEXT:   dpct::lapack::syhegvd_scratchpad_size<sycl::double2>(*handle, itype, jobz, uplo, n, lda, ldb, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsygvj | FileCheck %s -check-prefix=cusolverDnSsygvj
// cusolverDnSsygvj: CUDA API:
// cusolverDnSsygvj-NEXT:   cusolverDnSsygvj(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnSsygvj-NEXT:                    jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSsygvj-NEXT:                    n /*int*/, a /*float **/, lda /*int*/, b /*float **/,
// cusolverDnSsygvj-NEXT:                    ldb /*int*/, w /*float **/, buffer /*float **/,
// cusolverDnSsygvj-NEXT:                    buffer_size /*int*/, info /*int **/, params /*syevjInfo_t*/);
// cusolverDnSsygvj-NEXT: Is migrated to:
// cusolverDnSsygvj-NEXT:   dpct::lapack::syhegvd<float, float>(*handle, itype, jobz, uplo, n, a, lda, b, ldb, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsygvj | FileCheck %s -check-prefix=cusolverDnDsygvj
// cusolverDnDsygvj: CUDA API:
// cusolverDnDsygvj-NEXT:   cusolverDnDsygvj(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnDsygvj-NEXT:                    jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDsygvj-NEXT:                    n /*int*/, a /*double **/, lda /*int*/, b /*double **/,
// cusolverDnDsygvj-NEXT:                    ldb /*int*/, w /*double **/, buffer /*double **/,
// cusolverDnDsygvj-NEXT:                    buffer_size /*int*/, info /*int **/, params /*syevjInfo_t*/);
// cusolverDnDsygvj-NEXT: Is migrated to:
// cusolverDnDsygvj-NEXT:   dpct::lapack::syhegvd<double, double>(*handle, itype, jobz, uplo, n, a, lda, b, ldb, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnChegvj | FileCheck %s -check-prefix=cusolverDnChegvj
// cusolverDnChegvj: CUDA API:
// cusolverDnChegvj-NEXT:   cusolverDnChegvj(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnChegvj-NEXT:                    jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnChegvj-NEXT:                    n /*int*/, a /*cuComplex **/, lda /*int*/, b /*cuComplex **/,
// cusolverDnChegvj-NEXT:                    ldb /*int*/, w /*float **/, buffer /*cuComplex **/,
// cusolverDnChegvj-NEXT:                    buffer_size /*int*/, info /*int **/, params /*syevjInfo_t*/);
// cusolverDnChegvj-NEXT: Is migrated to:
// cusolverDnChegvj-NEXT:   dpct::lapack::syhegvd<sycl::float2, float>(*handle, itype, jobz, uplo, n, a, lda, b, ldb, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZhegvj | FileCheck %s -check-prefix=cusolverDnZhegvj
// cusolverDnZhegvj: CUDA API:
// cusolverDnZhegvj-NEXT:   cusolverDnZhegvj(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnZhegvj-NEXT:                    jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnZhegvj-NEXT:                    n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZhegvj-NEXT:                    b /*cuDoubleComplex **/, ldb /*int*/, w /*double **/,
// cusolverDnZhegvj-NEXT:                    buffer /*cuDoubleComplex **/, buffer_size /*int*/,
// cusolverDnZhegvj-NEXT:                    info /*int **/, params /*syevjInfo_t*/);
// cusolverDnZhegvj-NEXT: Is migrated to:
// cusolverDnZhegvj-NEXT:   dpct::lapack::syhegvd<sycl::double2, double>(*handle, itype, jobz, uplo, n, a, lda, b, ldb, w, buffer, buffer_size, info);
