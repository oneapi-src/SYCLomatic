// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsyevdx_bufferSize | FileCheck %s -check-prefix=cusolverDnSsyevdx_bufferSize
// cusolverDnSsyevdx_bufferSize: CUDA API:
// cusolverDnSsyevdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsyevdx_bufferSize-NEXT:   cusolverDnSsyevdx_bufferSize(
// cusolverDnSsyevdx_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnSsyevdx_bufferSize-NEXT:       range /*cusolverEigRange_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnSsyevdx_bufferSize-NEXT:       a /*const float **/, lda /*int*/, vl /*float*/, vu /*float*/, il /*int*/,
// cusolverDnSsyevdx_bufferSize-NEXT:       iu /*int*/, h_meig /*int **/, w /*const float **/,
// cusolverDnSsyevdx_bufferSize-NEXT:       &buffer_size /*int **/);
// cusolverDnSsyevdx_bufferSize-NEXT: Is migrated to:
// cusolverDnSsyevdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsyevdx_bufferSize-NEXT:   dpct::lapack::syheevx_scratchpad_size<float, float>(*handle, jobz, range, uplo, n, lda, vl, vu, il, iu, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsyevdx_bufferSize | FileCheck %s -check-prefix=cusolverDnDsyevdx_bufferSize
// cusolverDnDsyevdx_bufferSize: CUDA API:
// cusolverDnDsyevdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsyevdx_bufferSize-NEXT:   cusolverDnDsyevdx_bufferSize(
// cusolverDnDsyevdx_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnDsyevdx_bufferSize-NEXT:       range /*cusolverEigRange_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnDsyevdx_bufferSize-NEXT:       a /*const double **/, lda /*int*/, vl /*double*/, vu /*double*/,
// cusolverDnDsyevdx_bufferSize-NEXT:       il /*int*/, iu /*int*/, h_meig /*int **/, w /*const double **/,
// cusolverDnDsyevdx_bufferSize-NEXT:       &buffer_size /*int **/);
// cusolverDnDsyevdx_bufferSize-NEXT: Is migrated to:
// cusolverDnDsyevdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsyevdx_bufferSize-NEXT:   dpct::lapack::syheevx_scratchpad_size<double, double>(*handle, jobz, range, uplo, n, lda, vl, vu, il, iu, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCheevdx_bufferSize | FileCheck %s -check-prefix=cusolverDnCheevdx_bufferSize
// cusolverDnCheevdx_bufferSize: CUDA API:
// cusolverDnCheevdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnCheevdx_bufferSize-NEXT:   cusolverDnCheevdx_bufferSize(
// cusolverDnCheevdx_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnCheevdx_bufferSize-NEXT:       range /*cusolverEigRange_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnCheevdx_bufferSize-NEXT:       a /*const cuComplex **/, lda /*int*/, vl /*float*/, vu /*float*/,
// cusolverDnCheevdx_bufferSize-NEXT:       il /*int*/, iu /*int*/, h_meig /*int **/, w /*const float **/,
// cusolverDnCheevdx_bufferSize-NEXT:       &buffer_size /*int **/);
// cusolverDnCheevdx_bufferSize-NEXT: Is migrated to:
// cusolverDnCheevdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnCheevdx_bufferSize-NEXT:   dpct::lapack::syheevx_scratchpad_size<sycl::float2, float>(*handle, jobz, range, uplo, n, lda, vl, vu, il, iu, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZheevdx_bufferSize | FileCheck %s -check-prefix=cusolverDnZheevdx_bufferSize
// cusolverDnZheevdx_bufferSize: CUDA API:
// cusolverDnZheevdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnZheevdx_bufferSize-NEXT:   cusolverDnZheevdx_bufferSize(
// cusolverDnZheevdx_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnZheevdx_bufferSize-NEXT:       range /*cusolverEigRange_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZheevdx_bufferSize-NEXT:       a /*const cuDoubleComplex **/, lda /*int*/, vl /*double*/, vu /*double*/,
// cusolverDnZheevdx_bufferSize-NEXT:       il /*int*/, iu /*int*/, h_meig /*int **/, w /*const double **/,
// cusolverDnZheevdx_bufferSize-NEXT:       &buffer_size /*int **/);
// cusolverDnZheevdx_bufferSize-NEXT: Is migrated to:
// cusolverDnZheevdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnZheevdx_bufferSize-NEXT:   dpct::lapack::syheevx_scratchpad_size<sycl::double2, double>(*handle, jobz, range, uplo, n, lda, vl, vu, il, iu, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsyevdx | FileCheck %s -check-prefix=cusolverDnSsyevdx
// cusolverDnSsyevdx: CUDA API:
// cusolverDnSsyevdx-NEXT:   cusolverDnSsyevdx(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnSsyevdx-NEXT:                     range /*cusolverEigRange_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSsyevdx-NEXT:                     n /*int*/, a /*float **/, lda /*int*/, vl /*float*/,
// cusolverDnSsyevdx-NEXT:                     vu /*float*/, il /*int*/, iu /*int*/, h_meig /*int **/,
// cusolverDnSsyevdx-NEXT:                     w /*float **/, buffer /*float **/, buffer_size /*int*/,
// cusolverDnSsyevdx-NEXT:                     info /*int **/);
// cusolverDnSsyevdx-NEXT: Is migrated to:
// cusolverDnSsyevdx-NEXT:   dpct::lapack::syheevx<float, float>(*handle, jobz, range, uplo, n, a, lda, vl, vu, il, iu, h_meig, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsyevdx | FileCheck %s -check-prefix=cusolverDnDsyevdx
// cusolverDnDsyevdx: CUDA API:
// cusolverDnDsyevdx-NEXT:   cusolverDnDsyevdx(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnDsyevdx-NEXT:                     range /*cusolverEigRange_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDsyevdx-NEXT:                     n /*int*/, a /*double **/, lda /*int*/, vl /*double*/,
// cusolverDnDsyevdx-NEXT:                     vu /*double*/, il /*int*/, iu /*int*/, h_meig /*int **/,
// cusolverDnDsyevdx-NEXT:                     w /*double **/, buffer /*double **/, buffer_size /*int*/,
// cusolverDnDsyevdx-NEXT:                     info /*int **/);
// cusolverDnDsyevdx-NEXT: Is migrated to:
// cusolverDnDsyevdx-NEXT:   dpct::lapack::syheevx<double, double>(*handle, jobz, range, uplo, n, a, lda, vl, vu, il, iu, h_meig, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCheevdx | FileCheck %s -check-prefix=cusolverDnCheevdx
// cusolverDnCheevdx: CUDA API:
// cusolverDnCheevdx-NEXT:   cusolverDnCheevdx(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnCheevdx-NEXT:                     range /*cusolverEigRange_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnCheevdx-NEXT:                     n /*int*/, a /*cuComplex **/, lda /*int*/, vl /*float*/,
// cusolverDnCheevdx-NEXT:                     vu /*float*/, il /*int*/, iu /*int*/, h_meig /*int **/,
// cusolverDnCheevdx-NEXT:                     w /*float **/, buffer /*cuComplex **/, buffer_size /*int*/,
// cusolverDnCheevdx-NEXT:                     info /*int **/);
// cusolverDnCheevdx-NEXT: Is migrated to:
// cusolverDnCheevdx-NEXT:   dpct::lapack::syheevx<sycl::float2, float>(*handle, jobz, range, uplo, n, a, lda, vl, vu, il, iu, h_meig, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZheevdx | FileCheck %s -check-prefix=cusolverDnZheevdx
// cusolverDnZheevdx: CUDA API:
// cusolverDnZheevdx-NEXT:   cusolverDnZheevdx(
// cusolverDnZheevdx-NEXT:       handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
// cusolverDnZheevdx-NEXT:       range /*cusolverEigRange_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZheevdx-NEXT:       a /*cuDoubleComplex **/, lda /*int*/, vl /*double*/, vu /*double*/,
// cusolverDnZheevdx-NEXT:       il /*int*/, iu /*int*/, h_meig /*int **/, w /*double **/,
// cusolverDnZheevdx-NEXT:       buffer /*cuDoubleComplex **/, buffer_size /*int*/, info /*int **/);
// cusolverDnZheevdx-NEXT: Is migrated to:
// cusolverDnZheevdx-NEXT:   dpct::lapack::syheevx<sycl::double2, double>(*handle, jobz, range, uplo, n, a, lda, vl, vu, il, iu, h_meig, w, buffer, buffer_size, info);

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsygvdx_bufferSize | FileCheck %s -check-prefix=cusolverDnSsygvdx_bufferSize
// cusolverDnSsygvdx_bufferSize: CUDA API:
// cusolverDnSsygvdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsygvdx_bufferSize-NEXT:   cusolverDnSsygvdx_bufferSize(
// cusolverDnSsygvdx_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnSsygvdx_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
// cusolverDnSsygvdx_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int*/, a /*const float **/, lda /*int*/,
// cusolverDnSsygvdx_bufferSize-NEXT:       b /*const float **/, ldb /*int*/, vl /*float*/, vu /*float*/, il /*int*/,
// cusolverDnSsygvdx_bufferSize-NEXT:       iu /*int*/, h_meig /*int **/, w /*const float **/,
// cusolverDnSsygvdx_bufferSize-NEXT:       &buffer_size /*int **/);
// cusolverDnSsygvdx_bufferSize-NEXT: Is migrated to:
// cusolverDnSsygvdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnSsygvdx_bufferSize-NEXT:   dpct::lapack::syhegvx_scratchpad_size<float, float>(*handle, itype, jobz, range, uplo, n, lda, ldb, vl, vu, il, iu, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsygvdx_bufferSize | FileCheck %s -check-prefix=cusolverDnDsygvdx_bufferSize
// cusolverDnDsygvdx_bufferSize: CUDA API:
// cusolverDnDsygvdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsygvdx_bufferSize-NEXT:   cusolverDnDsygvdx_bufferSize(
// cusolverDnDsygvdx_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnDsygvdx_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
// cusolverDnDsygvdx_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int*/, a /*const double **/, lda /*int*/,
// cusolverDnDsygvdx_bufferSize-NEXT:       b /*const double **/, ldb /*int*/, vl /*double*/, vu /*double*/,
// cusolverDnDsygvdx_bufferSize-NEXT:       il /*int*/, iu /*int*/, h_meig /*int **/, w /*const double **/,
// cusolverDnDsygvdx_bufferSize-NEXT:       &buffer_size /*int **/);
// cusolverDnDsygvdx_bufferSize-NEXT: Is migrated to:
// cusolverDnDsygvdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnDsygvdx_bufferSize-NEXT:   dpct::lapack::syhegvx_scratchpad_size<double, double>(*handle, itype, jobz, range, uplo, n, lda, ldb, vl, vu, il, iu, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnChegvdx_bufferSize | FileCheck %s -check-prefix=cusolverDnChegvdx_bufferSize
// cusolverDnChegvdx_bufferSize: CUDA API:
// cusolverDnChegvdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnChegvdx_bufferSize-NEXT:   cusolverDnChegvdx_bufferSize(
// cusolverDnChegvdx_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnChegvdx_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
// cusolverDnChegvdx_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int*/, a /*const cuComplex **/,
// cusolverDnChegvdx_bufferSize-NEXT:       lda /*int*/, b /*const cuComplex **/, ldb /*int*/, vl /*float*/,
// cusolverDnChegvdx_bufferSize-NEXT:       vu /*float*/, il /*int*/, iu /*int*/, h_meig /*int **/,
// cusolverDnChegvdx_bufferSize-NEXT:       w /*const float **/, &buffer_size /*int **/);
// cusolverDnChegvdx_bufferSize-NEXT: Is migrated to:
// cusolverDnChegvdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnChegvdx_bufferSize-NEXT:   dpct::lapack::syhegvx_scratchpad_size<sycl::float2, float>(*handle, itype, jobz, range, uplo, n, lda, ldb, vl, vu, il, iu, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZhegvdx_bufferSize | FileCheck %s -check-prefix=cusolverDnZhegvdx_bufferSize
// cusolverDnZhegvdx_bufferSize: CUDA API:
// cusolverDnZhegvdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnZhegvdx_bufferSize-NEXT:   cusolverDnZhegvdx_bufferSize(
// cusolverDnZhegvdx_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnZhegvdx_bufferSize-NEXT:       jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
// cusolverDnZhegvdx_bufferSize-NEXT:       uplo /*cublasFillMode_t*/, n /*int*/, a /*const cuDoubleComplex **/,
// cusolverDnZhegvdx_bufferSize-NEXT:       lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/, vl /*double*/,
// cusolverDnZhegvdx_bufferSize-NEXT:       vu /*double*/, il /*int*/, iu /*int*/, h_meig /*int **/,
// cusolverDnZhegvdx_bufferSize-NEXT:       w /*const double **/, &buffer_size /*int **/);
// cusolverDnZhegvdx_bufferSize-NEXT: Is migrated to:
// cusolverDnZhegvdx_bufferSize-NEXT:   int buffer_size;
// cusolverDnZhegvdx_bufferSize-NEXT:   dpct::lapack::syhegvx_scratchpad_size<sycl::double2, double>(*handle, itype, jobz, range, uplo, n, lda, ldb, vl, vu, il, iu, &buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSsygvdx | FileCheck %s -check-prefix=cusolverDnSsygvdx
// cusolverDnSsygvdx: CUDA API:
// cusolverDnSsygvdx-NEXT:   cusolverDnSsygvdx(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnSsygvdx-NEXT:                     jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
// cusolverDnSsygvdx-NEXT:                     uplo /*cublasFillMode_t*/, n /*int*/, a /*float **/,
// cusolverDnSsygvdx-NEXT:                     lda /*int*/, b /*float **/, ldb /*int*/, vl /*float*/,
// cusolverDnSsygvdx-NEXT:                     vu /*float*/, il /*int*/, iu /*int*/, h_meig /*int **/,
// cusolverDnSsygvdx-NEXT:                     w /*float **/, buffer /*float **/, buffer_size /*int*/,
// cusolverDnSsygvdx-NEXT:                     info /*int **/);
// cusolverDnSsygvdx-NEXT: Is migrated to:
// cusolverDnSsygvdx-NEXT:   dpct::lapack::syhegvx<float, float>(*handle, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, h_meig, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDsygvdx | FileCheck %s -check-prefix=cusolverDnDsygvdx
// cusolverDnDsygvdx: CUDA API:
// cusolverDnDsygvdx-NEXT:   cusolverDnDsygvdx(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnDsygvdx-NEXT:                     jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
// cusolverDnDsygvdx-NEXT:                     uplo /*cublasFillMode_t*/, n /*int*/, a /*double **/,
// cusolverDnDsygvdx-NEXT:                     lda /*int*/, b /*double **/, ldb /*int*/, vl /*double*/,
// cusolverDnDsygvdx-NEXT:                     vu /*double*/, il /*int*/, iu /*int*/, h_meig /*int **/,
// cusolverDnDsygvdx-NEXT:                     w /*double **/, buffer /*double **/, buffer_size /*int*/,
// cusolverDnDsygvdx-NEXT:                     info /*int **/);
// cusolverDnDsygvdx-NEXT: Is migrated to:
// cusolverDnDsygvdx-NEXT:   dpct::lapack::syhegvx<double, double>(*handle, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, h_meig, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnChegvdx | FileCheck %s -check-prefix=cusolverDnChegvdx
// cusolverDnChegvdx: CUDA API:
// cusolverDnChegvdx-NEXT:   cusolverDnChegvdx(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnChegvdx-NEXT:                     jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
// cusolverDnChegvdx-NEXT:                     uplo /*cublasFillMode_t*/, n /*int*/, a /*cuComplex **/,
// cusolverDnChegvdx-NEXT:                     lda /*int*/, b /*cuComplex **/, ldb /*int*/, vl /*float*/,
// cusolverDnChegvdx-NEXT:                     vu /*float*/, il /*int*/, iu /*int*/, h_meig /*int **/,
// cusolverDnChegvdx-NEXT:                     w /*float **/, buffer /*cuComplex **/, buffer_size /*int*/,
// cusolverDnChegvdx-NEXT:                     info /*int **/);
// cusolverDnChegvdx-NEXT: Is migrated to:
// cusolverDnChegvdx-NEXT:   dpct::lapack::syhegvx<sycl::float2, float>(*handle, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, h_meig, w, buffer, buffer_size, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZhegvdx | FileCheck %s -check-prefix=cusolverDnZhegvdx
// cusolverDnZhegvdx: CUDA API:
// cusolverDnZhegvdx-NEXT:   cusolverDnZhegvdx(
// cusolverDnZhegvdx-NEXT:       handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
// cusolverDnZhegvdx-NEXT:       jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
// cusolverDnZhegvdx-NEXT:       uplo /*cublasFillMode_t*/, n /*int*/, a /*cuDoubleComplex **/,
// cusolverDnZhegvdx-NEXT:       lda /*int*/, b /*cuDoubleComplex **/, ldb /*int*/, vl /*double*/,
// cusolverDnZhegvdx-NEXT:       vu /*double*/, il /*int*/, iu /*int*/, h_meig /*int **/, w /*double **/,
// cusolverDnZhegvdx-NEXT:       buffer /*cuDoubleComplex **/, buffer_size /*int*/, info /*int **/);
// cusolverDnZhegvdx-NEXT: Is migrated to:
// cusolverDnZhegvdx-NEXT:   dpct::lapack::syhegvx<sycl::double2, double>(*handle, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, h_meig, w, buffer, buffer_size, info);

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXgeqrf | FileCheck %s -check-prefix=cusolverDnXgeqrf
// cusolverDnXgeqrf: CUDA API:
// cusolverDnXgeqrf-NEXT:   cusolverDnXgeqrf(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
// cusolverDnXgeqrf-NEXT:                    m /*int64_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
// cusolverDnXgeqrf-NEXT:                    a /* void **/, lda /*int64_t*/, tau_type /*cudaDataType*/,
// cusolverDnXgeqrf-NEXT:                    tau /* void **/, compute_type /*cudaDataType*/,
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
// cusolverDnXgesvd-NEXT:                    n /*int64_t*/, a_type /*cudaDataType*/, a /* void **/,
// cusolverDnXgesvd-NEXT:                    lda /*int64_t*/, s_type /*cudaDataType*/, s /* void **/,
// cusolverDnXgesvd-NEXT:                    u_type /*cudaDataType*/, u /* void **/, ldu /*int64_t*/,
// cusolverDnXgesvd-NEXT:                    vt_type /*cudaDataType*/, vt /* void **/, ldvt /*int64_t*/,
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
// cusolverDnXgetrf-NEXT:                    a /* void **/, lda /*int64_t*/, ipiv /*int64_t **/,
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
// cusolverDnXpotrf-NEXT:                    a_type /*cudaDataType*/, a /* void **/, lda /*int64_t*/,
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
// cusolverDnXsyevd-NEXT:                    n /*int64_t*/, a_type /*cudaDataType*/, a /* void **/,
// cusolverDnXsyevd-NEXT:                    lda /*int64_t*/, w_type /*cudaDataType*/, w /* void **/,
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
// cusolverDnXsyevdx-NEXT:       a /* void **/, lda /*int64_t*/, vl /*void **/, vu /*void **/,
// cusolverDnXsyevdx-NEXT:       il /*int64_t*/, iu /*int64_t*/, h_meig /*int64_t **/,
// cusolverDnXsyevdx-NEXT:       w_type /*cudaDataType*/, w /* void **/, compute_type /*cudaDataType*/,
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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnXtrtri | FileCheck %s -check-prefix=cusolverDnXtrtri
// cusolverDnXtrtri: CUDA API:
// cusolverDnXtrtri-NEXT:   cusolverDnXtrtri(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnXtrtri-NEXT:                    diag /*cublasDiagType_t*/, n /*int64_t*/,
// cusolverDnXtrtri-NEXT:                    a_type /*cudaDataType*/, a /* void **/, lda /*int64_t*/,
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
// cusolverDnXtrtri_bufferSize-NEXT:       a /*const void **/, lda /*int64_t*/, &device_buffer_size /*size_t **/,
// cusolverDnXtrtri_bufferSize-NEXT:       &host_buffer_size /*size_t **/);
// cusolverDnXtrtri_bufferSize-NEXT: Is migrated to:
// cusolverDnXtrtri_bufferSize-NEXT:   size_t device_buffer_size;
// cusolverDnXtrtri_bufferSize-NEXT:   size_t host_buffer_size;
// cusolverDnXtrtri_bufferSize-NEXT:   dpct::lapack::trtri_scratchpad_size(*handle, uplo, diag, n, a_type, lda, &device_buffer_size, &host_buffer_size);
