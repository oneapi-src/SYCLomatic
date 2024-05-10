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
