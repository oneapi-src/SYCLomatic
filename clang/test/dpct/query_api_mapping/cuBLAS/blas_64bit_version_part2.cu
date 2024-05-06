// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgbmv_64 | FileCheck %s -check-prefix=cublasZgbmv_64
// cublasZgbmv_64: CUDA API:
// cublasZgbmv_64-NEXT:   cublasZgbmv_64(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
// cublasZgbmv_64-NEXT:                  m /*int64_t*/, n /*int64_t*/, kl /*int64_t*/, ku /*int64_t*/,
// cublasZgbmv_64-NEXT:                  alpha /*const cuDoubleComplex **/,
// cublasZgbmv_64-NEXT:                  a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZgbmv_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZgbmv_64-NEXT:                  beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/,
// cublasZgbmv_64-NEXT:                  incy /*int64_t*/);
// cublasZgbmv_64-NEXT: Is migrated to:
// cublasZgbmv_64-NEXT:   oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), trans, m, n, kl, ku, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgeam_64 | FileCheck %s -check-prefix=cublasSgeam_64
// cublasSgeam_64: CUDA API:
// cublasSgeam_64-NEXT:   cublasSgeam_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasSgeam_64-NEXT:                  transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasSgeam_64-NEXT:                  alpha /*const float **/, a /*const float **/, lda /*int64_t*/,
// cublasSgeam_64-NEXT:                  beta /*const float **/, b /*const float **/, ldb /*int64_t*/,
// cublasSgeam_64-NEXT:                  c /*float **/, ldc /*int64_t*/);
// cublasSgeam_64-NEXT: Is migrated to:
// cublasSgeam_64-NEXT:   oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha, handle->get_queue()), a, lda, dpct::get_value(beta, handle->get_queue()), b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgeam_64 | FileCheck %s -check-prefix=cublasDgeam_64
// cublasDgeam_64: CUDA API:
// cublasDgeam_64-NEXT:   cublasDgeam_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasDgeam_64-NEXT:                  transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasDgeam_64-NEXT:                  alpha /*const double **/, a /*const double **/,
// cublasDgeam_64-NEXT:                  lda /*int64_t*/, beta /*const double **/, b /*const double **/,
// cublasDgeam_64-NEXT:                  ldb /*int64_t*/, c /*double **/, ldc /*int64_t*/);
// cublasDgeam_64-NEXT: Is migrated to:
// cublasDgeam_64-NEXT:   oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha, handle->get_queue()), a, lda, dpct::get_value(beta, handle->get_queue()), b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgeam_64 | FileCheck %s -check-prefix=cublasCgeam_64
// cublasCgeam_64: CUDA API:
// cublasCgeam_64-NEXT:   cublasCgeam_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasCgeam_64-NEXT:                  transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasCgeam_64-NEXT:                  alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCgeam_64-NEXT:                  lda /*int64_t*/, beta /*const cuComplex **/,
// cublasCgeam_64-NEXT:                  b /*const cuComplex **/, ldb /*int64_t*/, c /*cuComplex **/,
// cublasCgeam_64-NEXT:                  ldc /*int64_t*/);
// cublasCgeam_64-NEXT: Is migrated to:
// cublasCgeam_64-NEXT:   oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)b, ldb, (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgeam_64 | FileCheck %s -check-prefix=cublasZgeam_64
// cublasZgeam_64: CUDA API:
// cublasZgeam_64-NEXT:   cublasZgeam_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasZgeam_64-NEXT:                  transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasZgeam_64-NEXT:                  alpha /*const cuDoubleComplex **/,
// cublasZgeam_64-NEXT:                  a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZgeam_64-NEXT:                  beta /*const cuDoubleComplex **/,
// cublasZgeam_64-NEXT:                  b /*const cuDoubleComplex **/, ldb /*int64_t*/,
// cublasZgeam_64-NEXT:                  c /*cuDoubleComplex **/, ldc /*int64_t*/);
// cublasZgeam_64-NEXT: Is migrated to:
// cublasZgeam_64-NEXT:   oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)b, ldb, (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgemm_64 | FileCheck %s -check-prefix=cublasSgemm_64
// cublasSgemm_64: CUDA API:
// cublasSgemm_64-NEXT:   cublasSgemm_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasSgemm_64-NEXT:                  transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasSgemm_64-NEXT:                  k /*int64_t*/, alpha /*const float **/, a /*const float **/,
// cublasSgemm_64-NEXT:                  lda /*int64_t*/, b /*const float **/, ldb /*int64_t*/,
// cublasSgemm_64-NEXT:                  beta /*const float **/, c /*float **/, ldc /*int64_t*/);
// cublasSgemm_64-NEXT: Is migrated to:
// cublasSgemm_64-NEXT:   oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgemm_64 | FileCheck %s -check-prefix=cublasDgemm_64
// cublasDgemm_64: CUDA API:
// cublasDgemm_64-NEXT:   cublasDgemm_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasDgemm_64-NEXT:                  transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasDgemm_64-NEXT:                  k /*int64_t*/, alpha /*const double **/, a /*const double **/,
// cublasDgemm_64-NEXT:                  lda /*int64_t*/, b /*const double **/, ldb /*int64_t*/,
// cublasDgemm_64-NEXT:                  beta /*const double **/, c /*double **/, ldc /*int64_t*/);
// cublasDgemm_64-NEXT: Is migrated to:
// cublasDgemm_64-NEXT:   oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemm_64 | FileCheck %s -check-prefix=cublasCgemm_64
// cublasCgemm_64: CUDA API:
// cublasCgemm_64-NEXT:   cublasCgemm_64(
// cublasCgemm_64-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasCgemm_64-NEXT:       transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasCgemm_64-NEXT:       alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int64_t*/,
// cublasCgemm_64-NEXT:       b /*const cuComplex **/, ldb /*int64_t*/, beta /*const cuComplex **/,
// cublasCgemm_64-NEXT:       c /*cuComplex **/, ldc /*int64_t*/);
// cublasCgemm_64-NEXT: Is migrated to:
// cublasCgemm_64-NEXT:   oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgemm_64 | FileCheck %s -check-prefix=cublasZgemm_64
// cublasZgemm_64: CUDA API:
// cublasZgemm_64-NEXT:   cublasZgemm_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasZgemm_64-NEXT:                  transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasZgemm_64-NEXT:                  k /*int64_t*/, alpha /*const cuDoubleComplex **/,
// cublasZgemm_64-NEXT:                  a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZgemm_64-NEXT:                  b /*const cuDoubleComplex **/, ldb /*int64_t*/,
// cublasZgemm_64-NEXT:                  beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZgemm_64-NEXT:                  ldc /*int64_t*/);
// cublasZgemm_64-NEXT: Is migrated to:
// cublasZgemm_64-NEXT:   oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgemv_64 | FileCheck %s -check-prefix=cublasSgemv_64
// cublasSgemv_64: CUDA API:
// cublasSgemv_64-NEXT:   cublasSgemv_64(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
// cublasSgemv_64-NEXT:                  m /*int64_t*/, n /*int64_t*/, alpha /*const float **/,
// cublasSgemv_64-NEXT:                  a /*const float **/, lda /*int64_t*/, x /*const float **/,
// cublasSgemv_64-NEXT:                  incx /*int64_t*/, beta /*const float **/, y /*float **/,
// cublasSgemv_64-NEXT:                  incy /*int64_t*/);
// cublasSgemv_64-NEXT: Is migrated to:
// cublasSgemv_64-NEXT:   oneapi::mkl::blas::column_major::gemv(handle->get_queue(), trans, m, n, dpct::get_value(alpha, handle->get_queue()), a, lda, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgemv_64 | FileCheck %s -check-prefix=cublasDgemv_64
// cublasDgemv_64: CUDA API:
// cublasDgemv_64-NEXT:   cublasDgemv_64(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
// cublasDgemv_64-NEXT:                  m /*int64_t*/, n /*int64_t*/, alpha /*const double **/,
// cublasDgemv_64-NEXT:                  a /*const double **/, lda /*int64_t*/, x /*const double **/,
// cublasDgemv_64-NEXT:                  incx /*int64_t*/, beta /*const double **/, y /*double **/,
// cublasDgemv_64-NEXT:                  incy /*int64_t*/);
// cublasDgemv_64-NEXT: Is migrated to:
// cublasDgemv_64-NEXT:   oneapi::mkl::blas::column_major::gemv(handle->get_queue(), trans, m, n, dpct::get_value(alpha, handle->get_queue()), a, lda, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemv_64 | FileCheck %s -check-prefix=cublasCgemv_64
// cublasCgemv_64: CUDA API:
// cublasCgemv_64-NEXT:   cublasCgemv_64(
// cublasCgemv_64-NEXT:       handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int64_t*/,
// cublasCgemv_64-NEXT:       n /*int64_t*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCgemv_64-NEXT:       lda /*int64_t*/, x /*const cuComplex **/, incx /*int64_t*/,
// cublasCgemv_64-NEXT:       beta /*const cuComplex **/, y /*cuComplex **/, incy /*int64_t*/);
// cublasCgemv_64-NEXT: Is migrated to:
// cublasCgemv_64-NEXT:   oneapi::mkl::blas::column_major::gemv(handle->get_queue(), trans, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgemv_64 | FileCheck %s -check-prefix=cublasZgemv_64
// cublasZgemv_64: CUDA API:
// cublasZgemv_64-NEXT:   cublasZgemv_64(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
// cublasZgemv_64-NEXT:                  m /*int64_t*/, n /*int64_t*/,
// cublasZgemv_64-NEXT:                  alpha /*const cuDoubleComplex **/,
// cublasZgemv_64-NEXT:                  a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZgemv_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZgemv_64-NEXT:                  beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/,
// cublasZgemv_64-NEXT:                  incy /*int64_t*/);
// cublasZgemv_64-NEXT: Is migrated to:
// cublasZgemv_64-NEXT:   oneapi::mkl::blas::column_major::gemv(handle->get_queue(), trans, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChemm_64 | FileCheck %s -check-prefix=cublasChemm_64
// cublasChemm_64: CUDA API:
// cublasChemm_64-NEXT:   cublasChemm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasChemm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasChemm_64-NEXT:                  alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasChemm_64-NEXT:                  lda /*int64_t*/, b /*const cuComplex **/, ldb /*int64_t*/,
// cublasChemm_64-NEXT:                  beta /*const cuComplex **/, c /*cuComplex **/,
// cublasChemm_64-NEXT:                  ldc /*int64_t*/);
// cublasChemm_64-NEXT: Is migrated to:
// cublasChemm_64-NEXT:   oneapi::mkl::blas::column_major::hemm(handle->get_queue(), left_right, upper_lower, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhemm_64 | FileCheck %s -check-prefix=cublasZhemm_64
// cublasZhemm_64: CUDA API:
// cublasZhemm_64-NEXT:   cublasZhemm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZhemm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasZhemm_64-NEXT:                  alpha /*const cuDoubleComplex **/,
// cublasZhemm_64-NEXT:                  a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZhemm_64-NEXT:                  b /*const cuDoubleComplex **/, ldb /*int64_t*/,
// cublasZhemm_64-NEXT:                  beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZhemm_64-NEXT:                  ldc /*int64_t*/);
// cublasZhemm_64-NEXT: Is migrated to:
// cublasZhemm_64-NEXT:   oneapi::mkl::blas::column_major::hemm(handle->get_queue(), left_right, upper_lower, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCher2k_64 | FileCheck %s -check-prefix=cublasCher2k_64
// cublasCher2k_64: CUDA API:
// cublasCher2k_64-NEXT:   cublasCher2k_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCher2k_64-NEXT:                   trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasCher2k_64-NEXT:                   alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCher2k_64-NEXT:                   lda /*int64_t*/, b /*const cuComplex **/, ldb /*int64_t*/,
// cublasCher2k_64-NEXT:                   beta /*const float **/, c /*cuComplex **/, ldc /*int64_t*/);
// cublasCher2k_64-NEXT: Is migrated to:
// cublasCher2k_64-NEXT:   oneapi::mkl::blas::column_major::her2k(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZher2k_64 | FileCheck %s -check-prefix=cublasZher2k_64
// cublasZher2k_64: CUDA API:
// cublasZher2k_64-NEXT:   cublasZher2k_64(
// cublasZher2k_64-NEXT:       handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZher2k_64-NEXT:       trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasZher2k_64-NEXT:       alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZher2k_64-NEXT:       lda /*int64_t*/, b /*const cuDoubleComplex **/, ldb /*int64_t*/,
// cublasZher2k_64-NEXT:       beta /*const double **/, c /*cuDoubleComplex **/, ldc /*int64_t*/);
// cublasZher2k_64-NEXT: Is migrated to:
// cublasZher2k_64-NEXT:   oneapi::mkl::blas::column_major::her2k(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCherk_64 | FileCheck %s -check-prefix=cublasCherk_64
// cublasCherk_64: CUDA API:
// cublasCherk_64-NEXT:   cublasCherk_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCherk_64-NEXT:                  trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasCherk_64-NEXT:                  alpha /*const float **/, a /*const cuComplex **/,
// cublasCherk_64-NEXT:                  lda /*int64_t*/, beta /*const float **/, c /*cuComplex **/,
// cublasCherk_64-NEXT:                  ldc /*int64_t*/);
// cublasCherk_64-NEXT: Is migrated to:
// cublasCherk_64-NEXT:   oneapi::mkl::blas::column_major::herk(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZherk_64 | FileCheck %s -check-prefix=cublasZherk_64
// cublasZherk_64: CUDA API:
// cublasZherk_64-NEXT:   cublasZherk_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZherk_64-NEXT:                  trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasZherk_64-NEXT:                  alpha /*const double **/, a /*const cuDoubleComplex **/,
// cublasZherk_64-NEXT:                  lda /*int64_t*/, beta /*const double **/,
// cublasZherk_64-NEXT:                  c /*cuDoubleComplex **/, ldc /*int64_t*/);
// cublasZherk_64-NEXT: Is migrated to:
// cublasZherk_64-NEXT:   oneapi::mkl::blas::column_major::herk(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc);
