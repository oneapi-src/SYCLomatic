// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCherkx_64 | FileCheck %s -check-prefix=cublasCherkx_64
// cublasCherkx_64: CUDA API:
// cublasCherkx_64-NEXT:   cublasCherkx_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCherkx_64-NEXT:                   trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasCherkx_64-NEXT:                   alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCherkx_64-NEXT:                   lda /*int64_t*/, b /*const cuComplex **/, ldb /*int64_t*/,
// cublasCherkx_64-NEXT:                   beta /*const float **/, c /*cuComplex **/, ldc /*int64_t*/);
// cublasCherkx_64-NEXT: Is migrated to:
// cublasCherkx_64-NEXT:   dpct::blas::herk(handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZherkx_64 | FileCheck %s -check-prefix=cublasZherkx_64
// cublasZherkx_64: CUDA API:
// cublasZherkx_64-NEXT:   cublasZherkx_64(
// cublasZherkx_64-NEXT:       handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZherkx_64-NEXT:       trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasZherkx_64-NEXT:       alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZherkx_64-NEXT:       lda /*int64_t*/, b /*const cuDoubleComplex **/, ldb /*int64_t*/,
// cublasZherkx_64-NEXT:       beta /*const double **/, c /*cuDoubleComplex **/, ldc /*int64_t*/);
// cublasZherkx_64-NEXT: Is migrated to:
// cublasZherkx_64-NEXT:   dpct::blas::herk(handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChpmv_64 | FileCheck %s -check-prefix=cublasChpmv_64
// cublasChpmv_64: CUDA API:
// cublasChpmv_64-NEXT:   cublasChpmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChpmv_64-NEXT:                  n /*int64_t*/, alpha /*const cuComplex **/,
// cublasChpmv_64-NEXT:                  a /*const cuComplex **/, x /*const cuComplex **/,
// cublasChpmv_64-NEXT:                  incx /*int64_t*/, beta /*const cuComplex **/,
// cublasChpmv_64-NEXT:                  y /*cuComplex **/, incy /*int64_t*/);
// cublasChpmv_64-NEXT: Is migrated to:
// cublasChpmv_64-NEXT:   oneapi::mkl::blas::column_major::hpmv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, (std::complex<float>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhpmv_64 | FileCheck %s -check-prefix=cublasZhpmv_64
// cublasZhpmv_64: CUDA API:
// cublasZhpmv_64-NEXT:   cublasZhpmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZhpmv_64-NEXT:                  n /*int64_t*/, alpha /*const cuDoubleComplex **/,
// cublasZhpmv_64-NEXT:                  a /*const cuDoubleComplex **/, x /*const cuDoubleComplex **/,
// cublasZhpmv_64-NEXT:                  incx /*int64_t*/, beta /*const cuDoubleComplex **/,
// cublasZhpmv_64-NEXT:                  y /*cuDoubleComplex **/, incy /*int64_t*/);
// cublasZhpmv_64-NEXT: Is migrated to:
// cublasZhpmv_64-NEXT:   oneapi::mkl::blas::column_major::hpmv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, (std::complex<double>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChpr2_64 | FileCheck %s -check-prefix=cublasChpr2_64
// cublasChpr2_64: CUDA API:
// cublasChpr2_64-NEXT:   cublasChpr2_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChpr2_64-NEXT:                  n /*int64_t*/, alpha /*const cuComplex **/,
// cublasChpr2_64-NEXT:                  x /*const cuComplex **/, incx /*int64_t*/,
// cublasChpr2_64-NEXT:                  y /*const cuComplex **/, incy /*int64_t*/, a /*cuComplex **/);
// cublasChpr2_64-NEXT: Is migrated to:
// cublasChpr2_64-NEXT:   oneapi::mkl::blas::column_major::hpr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhpr2_64 | FileCheck %s -check-prefix=cublasZhpr2_64
// cublasZhpr2_64: CUDA API:
// cublasZhpr2_64-NEXT:   cublasZhpr2_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZhpr2_64-NEXT:                  n /*int64_t*/, alpha /*const cuDoubleComplex **/,
// cublasZhpr2_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZhpr2_64-NEXT:                  y /*const cuDoubleComplex **/, incy /*int64_t*/,
// cublasZhpr2_64-NEXT:                  a /*cuDoubleComplex **/);
// cublasZhpr2_64-NEXT: Is migrated to:
// cublasZhpr2_64-NEXT:   oneapi::mkl::blas::column_major::hpr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChpr_64 | FileCheck %s -check-prefix=cublasChpr_64
// cublasChpr_64: CUDA API:
// cublasChpr_64-NEXT:   cublasChpr_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChpr_64-NEXT:                 n /*int64_t*/, alpha /*const float **/, x /*const cuComplex **/,
// cublasChpr_64-NEXT:                 incx /*int64_t*/, a /*cuComplex **/);
// cublasChpr_64-NEXT: Is migrated to:
// cublasChpr_64-NEXT:   oneapi::mkl::blas::column_major::hpr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhpr_64 | FileCheck %s -check-prefix=cublasZhpr_64
// cublasZhpr_64: CUDA API:
// cublasZhpr_64-NEXT:   cublasZhpr_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZhpr_64-NEXT:                 n /*int64_t*/, alpha /*const double **/,
// cublasZhpr_64-NEXT:                 x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZhpr_64-NEXT:                 a /*cuDoubleComplex **/);
// cublasZhpr_64-NEXT: Is migrated to:
// cublasZhpr_64-NEXT:   oneapi::mkl::blas::column_major::hpr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCopyEx_64 | FileCheck %s -check-prefix=cublasCopyEx_64
// cublasCopyEx_64: CUDA API:
// cublasCopyEx_64-NEXT:   cublasCopyEx_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const void **/,
// cublasCopyEx_64-NEXT:                   x_type /*cudaDataType*/, incx /*int64_t*/, y /*void **/,
// cublasCopyEx_64-NEXT:                   y_type /*cudaDataType*/, incy /*int64_t*/);
// cublasCopyEx_64-NEXT: Is migrated to:
// cublasCopyEx_64-NEXT:   dpct::blas::copy(handle, n, x, x_type, incx, y, y_type, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsymv_64 | FileCheck %s -check-prefix=cublasCsymv_64
// cublasCsymv_64: CUDA API:
// cublasCsymv_64-NEXT:   cublasCsymv_64(
// cublasCsymv_64-NEXT:       handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsymv_64-NEXT:       n /*int64_t*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCsymv_64-NEXT:       lda /*int64_t*/, x /*const cuComplex **/, incx /*int64_t*/,
// cublasCsymv_64-NEXT:       beta /*const cuComplex **/, y /*cuComplex **/, incy /*int64_t*/);
// cublasCsymv_64-NEXT: Is migrated to:
// cublasCsymv_64-NEXT:   oneapi::mkl::blas::column_major::symv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsymv_64 | FileCheck %s -check-prefix=cublasDsymv_64
// cublasDsymv_64: CUDA API:
// cublasDsymv_64-NEXT:   cublasDsymv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsymv_64-NEXT:                  n /*int64_t*/, alpha /*const double **/, a /*const double **/,
// cublasDsymv_64-NEXT:                  lda /*int64_t*/, x /*const double **/, incx /*int64_t*/,
// cublasDsymv_64-NEXT:                  beta /*const double **/, y /*double **/, incy /*int64_t*/);
// cublasDsymv_64-NEXT: Is migrated to:
// cublasDsymv_64-NEXT:   oneapi::mkl::blas::column_major::symv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), a, lda, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsymv_64 | FileCheck %s -check-prefix=cublasSsymv_64
// cublasSsymv_64: CUDA API:
// cublasSsymv_64-NEXT:   cublasSsymv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsymv_64-NEXT:                  n /*int64_t*/, alpha /*const float **/, a /*const float **/,
// cublasSsymv_64-NEXT:                  lda /*int64_t*/, x /*const float **/, incx /*int64_t*/,
// cublasSsymv_64-NEXT:                  beta /*const float **/, y /*float **/, incy /*int64_t*/);
// cublasSsymv_64-NEXT: Is migrated to:
// cublasSsymv_64-NEXT:   oneapi::mkl::blas::column_major::symv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), a, lda, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsymv_64 | FileCheck %s -check-prefix=cublasZsymv_64
// cublasZsymv_64: CUDA API:
// cublasZsymv_64-NEXT:   cublasZsymv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsymv_64-NEXT:                  n /*int64_t*/, alpha /*const cuDoubleComplex **/,
// cublasZsymv_64-NEXT:                  a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZsymv_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZsymv_64-NEXT:                  beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/,
// cublasZsymv_64-NEXT:                  incy /*int64_t*/);
// cublasZsymv_64-NEXT: Is migrated to:
// cublasZsymv_64-NEXT:   oneapi::mkl::blas::column_major::symv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyr2_64 | FileCheck %s -check-prefix=cublasCsyr2_64
// cublasCsyr2_64: CUDA API:
// cublasCsyr2_64-NEXT:   cublasCsyr2_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyr2_64-NEXT:                  n /*int64_t*/, alpha /*const cuComplex **/,
// cublasCsyr2_64-NEXT:                  x /*const cuComplex **/, incx /*int64_t*/,
// cublasCsyr2_64-NEXT:                  y /*const cuComplex **/, incy /*int64_t*/, a /*cuComplex **/,
// cublasCsyr2_64-NEXT:                  lda /*int64_t*/);
// cublasCsyr2_64-NEXT: Is migrated to:
// cublasCsyr2_64-NEXT:   oneapi::mkl::blas::column_major::syr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyr2_64 | FileCheck %s -check-prefix=cublasDsyr2_64
// cublasDsyr2_64: CUDA API:
// cublasDsyr2_64-NEXT:   cublasDsyr2_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyr2_64-NEXT:                  n /*int64_t*/, alpha /*const double **/, x /*const double **/,
// cublasDsyr2_64-NEXT:                  incx /*int64_t*/, y /*const double **/, incy /*int64_t*/,
// cublasDsyr2_64-NEXT:                  a /*double **/, lda /*int64_t*/);
// cublasDsyr2_64-NEXT: Is migrated to:
// cublasDsyr2_64-NEXT:   oneapi::mkl::blas::column_major::syr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, y, incy, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsyr2_64 | FileCheck %s -check-prefix=cublasSsyr2_64
// cublasSsyr2_64: CUDA API:
// cublasSsyr2_64-NEXT:   cublasSsyr2_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsyr2_64-NEXT:                  n /*int64_t*/, alpha /*const float **/, x /*const float **/,
// cublasSsyr2_64-NEXT:                  incx /*int64_t*/, y /*const float **/, incy /*int64_t*/,
// cublasSsyr2_64-NEXT:                  a /*float **/, lda /*int64_t*/);
// cublasSsyr2_64-NEXT: Is migrated to:
// cublasSsyr2_64-NEXT:   oneapi::mkl::blas::column_major::syr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, y, incy, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyr2_64 | FileCheck %s -check-prefix=cublasZsyr2_64
// cublasZsyr2_64: CUDA API:
// cublasZsyr2_64-NEXT:   cublasZsyr2_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyr2_64-NEXT:                  n /*int64_t*/, alpha /*const cuDoubleComplex **/,
// cublasZsyr2_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZsyr2_64-NEXT:                  y /*const cuDoubleComplex **/, incy /*int64_t*/,
// cublasZsyr2_64-NEXT:                  a /*cuDoubleComplex **/, lda /*int64_t*/);
// cublasZsyr2_64-NEXT: Is migrated to:
// cublasZsyr2_64-NEXT:   oneapi::mkl::blas::column_major::syr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyr_64 | FileCheck %s -check-prefix=cublasCsyr_64
// cublasCsyr_64: CUDA API:
// cublasCsyr_64-NEXT:   cublasCsyr_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyr_64-NEXT:                 n /*int64_t*/, alpha /*const cuComplex **/,
// cublasCsyr_64-NEXT:                 x /*const cuComplex **/, incx /*int64_t*/, a /*cuComplex **/,
// cublasCsyr_64-NEXT:                 lda /*int64_t*/);
// cublasCsyr_64-NEXT: Is migrated to:
// cublasCsyr_64-NEXT:   oneapi::mkl::blas::column_major::syr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyr_64 | FileCheck %s -check-prefix=cublasDsyr_64
// cublasDsyr_64: CUDA API:
// cublasDsyr_64-NEXT:   cublasDsyr_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyr_64-NEXT:                 n /*int64_t*/, alpha /*const double **/, x /*const double **/,
// cublasDsyr_64-NEXT:                 incx /*int64_t*/, a /*double **/, lda /*int64_t*/);
// cublasDsyr_64-NEXT: Is migrated to:
// cublasDsyr_64-NEXT:   oneapi::mkl::blas::column_major::syr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsyr_64 | FileCheck %s -check-prefix=cublasSsyr_64
// cublasSsyr_64: CUDA API:
// cublasSsyr_64-NEXT:   cublasSsyr_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsyr_64-NEXT:                 n /*int64_t*/, alpha /*const float **/, x /*const float **/,
// cublasSsyr_64-NEXT:                 incx /*int64_t*/, a /*float **/, lda /*int64_t*/);
// cublasSsyr_64-NEXT: Is migrated to:
// cublasSsyr_64-NEXT:   oneapi::mkl::blas::column_major::syr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, a, lda);
