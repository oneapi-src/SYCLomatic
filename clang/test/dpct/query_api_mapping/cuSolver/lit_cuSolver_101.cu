// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSpotri_bufferSize | FileCheck %s -check-prefix=cusolverDnSpotri_bufferSize
// cusolverDnSpotri_bufferSize: CUDA API:
// cusolverDnSpotri_bufferSize-NEXT:   int buffer_size;
// cusolverDnSpotri_bufferSize-NEXT:   cusolverDnSpotri_bufferSize(
// cusolverDnSpotri_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnSpotri_bufferSize-NEXT:       a /*float **/, lda /*int*/, &buffer_size /*int **/);
// cusolverDnSpotri_bufferSize-NEXT: Is migrated to:
// cusolverDnSpotri_bufferSize-NEXT:   int buffer_size;
// cusolverDnSpotri_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::potri_scratchpad_size<float>(
// cusolverDnSpotri_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*float **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDpotri_bufferSize | FileCheck %s -check-prefix=cusolverDnDpotri_bufferSize
// cusolverDnDpotri_bufferSize: CUDA API:
// cusolverDnDpotri_bufferSize-NEXT:   int buffer_size;
// cusolverDnDpotri_bufferSize-NEXT:   cusolverDnDpotri_bufferSize(
// cusolverDnDpotri_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnDpotri_bufferSize-NEXT:       a /*double **/, lda /*int*/, &buffer_size /*int **/);
// cusolverDnDpotri_bufferSize-NEXT: Is migrated to:
// cusolverDnDpotri_bufferSize-NEXT:   int buffer_size;
// cusolverDnDpotri_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::potri_scratchpad_size<double>(
// cusolverDnDpotri_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*double **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCpotri_bufferSize | FileCheck %s -check-prefix=cusolverDnCpotri_bufferSize
// cusolverDnCpotri_bufferSize: CUDA API:
// cusolverDnCpotri_bufferSize-NEXT:   int buffer_size;
// cusolverDnCpotri_bufferSize-NEXT:   cusolverDnCpotri_bufferSize(
// cusolverDnCpotri_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnCpotri_bufferSize-NEXT:       a /*cuComplex **/, lda /*int*/, &buffer_size /*int **/);
// cusolverDnCpotri_bufferSize-NEXT: Is migrated to:
// cusolverDnCpotri_bufferSize-NEXT:   int buffer_size;
// cusolverDnCpotri_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::potri_scratchpad_size<std::complex<float>>(
// cusolverDnCpotri_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*cuComplex **/, lda /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZpotri_bufferSize | FileCheck %s -check-prefix=cusolverDnZpotri_bufferSize
// cusolverDnZpotri_bufferSize: CUDA API:
// cusolverDnZpotri_bufferSize-NEXT:   int buffer_size;
// cusolverDnZpotri_bufferSize-NEXT:   cusolverDnZpotri_bufferSize(
// cusolverDnZpotri_bufferSize-NEXT:       handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
// cusolverDnZpotri_bufferSize-NEXT:       a /*cuDoubleComplex **/, lda /*int*/, &buffer_size /*int **/);
// cusolverDnZpotri_bufferSize-NEXT: Is migrated to:
// cusolverDnZpotri_bufferSize-NEXT:   int buffer_size;
// cusolverDnZpotri_bufferSize-NEXT:   *(&buffer_size) = oneapi::mkl::lapack::potri_scratchpad_size<std::complex<double>>(
// cusolverDnZpotri_bufferSize-NEXT:       *handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*cuDoubleComplex **/, lda /*int **/);

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnSpotri | FileCheck %s -check-prefix=cusolverDnSpotri
// cusolverDnSpotri: CUDA API:
// cusolverDnSpotri-NEXT:   cusolverDnSpotri(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSpotri-NEXT:                    n /*int*/, a /*float **/, lda /*int*/, buffer /*float **/,
// cusolverDnSpotri-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnSpotri-NEXT: Is migrated to:
// cusolverDnSpotri-NEXT:   oneapi::mkl::lapack::potri(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnSpotri-NEXT:                    n /*int*/, (float*)a /*float **/, lda /*int*/, (float*)buffer /*float **/,
// cusolverDnSpotri-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnDpotri | FileCheck %s -check-prefix=cusolverDnDpotri
// cusolverDnDpotri: CUDA API:
// cusolverDnDpotri-NEXT:   cusolverDnDpotri(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDpotri-NEXT:                    n /*int*/, a /*double **/, lda /*int*/, buffer /*double **/,
// cusolverDnDpotri-NEXT:                    buffer_size /*int*/, info /*int **/);
// cusolverDnDpotri-NEXT: Is migrated to:
// cusolverDnDpotri-NEXT:   oneapi::mkl::lapack::potri(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnDpotri-NEXT:                    n /*int*/, (double*)a /*double **/, lda /*int*/, (double*)buffer /*double **/,
// cusolverDnDpotri-NEXT:                    buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnCpotri | FileCheck %s -check-prefix=cusolverDnCpotri
// cusolverDnCpotri: CUDA API:
// cusolverDnCpotri-NEXT:   cusolverDnCpotri(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnCpotri-NEXT:                    n /*int*/, a /*cuComplex **/, lda /*int*/,
// cusolverDnCpotri-NEXT:                    buffer /*cuComplex **/, buffer_size /*int*/, info /*int **/);
// cusolverDnCpotri-NEXT: Is migrated to:
// cusolverDnCpotri-NEXT:   oneapi::mkl::lapack::potri(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnCpotri-NEXT:                    n /*int*/, (std::complex<float>*)a /*cuComplex **/, lda /*int*/,
// cusolverDnCpotri-NEXT:                    (std::complex<float>*)buffer /*cuComplex **/, buffer_size /*int **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusolverDnZpotri | FileCheck %s -check-prefix=cusolverDnZpotri
// cusolverDnZpotri: CUDA API:
// cusolverDnZpotri-NEXT:   cusolverDnZpotri(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnZpotri-NEXT:                    n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZpotri-NEXT:                    buffer /*cuDoubleComplex **/, buffer_size /*int*/,
// cusolverDnZpotri-NEXT:                    info /*int **/);
// cusolverDnZpotri-NEXT: Is migrated to:
// cusolverDnZpotri-NEXT:   oneapi::mkl::lapack::potri(*handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
// cusolverDnZpotri-NEXT:                    n /*int*/, (std::complex<double>*)a /*cuDoubleComplex **/, lda /*int*/,
// cusolverDnZpotri-NEXT:                    (std::complex<double>*)buffer /*cuDoubleComplex **/, buffer_size /*int **/);

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
