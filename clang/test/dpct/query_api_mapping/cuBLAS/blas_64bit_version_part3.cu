// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSrot_64 | FileCheck %s -check-prefix=cublasSrot_64
// cublasSrot_64: CUDA API:
// cublasSrot_64-NEXT:   cublasSrot_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*float **/,
// cublasSrot_64-NEXT:                 incx /*int64_t*/, y /*float **/, incy /*int64_t*/,
// cublasSrot_64-NEXT:                 c /*const float **/, s /*const float **/);
// cublasSrot_64-NEXT: Is migrated to:
// cublasSrot_64-NEXT:   oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, x, incx, y, incy, dpct::get_value(c, handle->get_queue()), dpct::get_value(s, handle->get_queue()));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDrot_64 | FileCheck %s -check-prefix=cublasDrot_64
// cublasDrot_64: CUDA API:
// cublasDrot_64-NEXT:   cublasDrot_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*double **/,
// cublasDrot_64-NEXT:                 incx /*int64_t*/, y /*double **/, incy /*int64_t*/,
// cublasDrot_64-NEXT:                 c /*const double **/, s /*const double **/);
// cublasDrot_64-NEXT: Is migrated to:
// cublasDrot_64-NEXT:   oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, x, incx, y, incy, dpct::get_value(c, handle->get_queue()), dpct::get_value(s, handle->get_queue()));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCrot_64 | FileCheck %s -check-prefix=cublasCrot_64
// cublasCrot_64: CUDA API:
// cublasCrot_64-NEXT:   cublasCrot_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*cuComplex **/,
// cublasCrot_64-NEXT:                 incx /*int64_t*/, y /*cuComplex **/, incy /*int64_t*/,
// cublasCrot_64-NEXT:                 c /*const float **/, s /*const cuComplex **/);
// cublasCrot_64-NEXT: Is migrated to:
// cublasCrot_64-NEXT:   oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, dpct::get_value(c, handle->get_queue()), dpct::get_value(s, handle->get_queue()));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsrot_64 | FileCheck %s -check-prefix=cublasCsrot_64
// cublasCsrot_64: CUDA API:
// cublasCsrot_64-NEXT:   cublasCsrot_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*cuComplex **/,
// cublasCsrot_64-NEXT:                  incx /*int64_t*/, y /*cuComplex **/, incy /*int64_t*/,
// cublasCsrot_64-NEXT:                  c /*const float **/, s /*const float **/);
// cublasCsrot_64-NEXT: Is migrated to:
// cublasCsrot_64-NEXT:   oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, dpct::get_value(c, handle->get_queue()), dpct::get_value(s, handle->get_queue()));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZrot_64 | FileCheck %s -check-prefix=cublasZrot_64
// cublasZrot_64: CUDA API:
// cublasZrot_64-NEXT:   cublasZrot_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasZrot_64-NEXT:                 x /*cuDoubleComplex **/, incx /*int64_t*/,
// cublasZrot_64-NEXT:                 y /*cuDoubleComplex **/, incy /*int64_t*/, c /*const double **/,
// cublasZrot_64-NEXT:                 s /*const cuDoubleComplex **/);
// cublasZrot_64-NEXT: Is migrated to:
// cublasZrot_64-NEXT:   oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, dpct::get_value(c, handle->get_queue()), dpct::get_value(s, handle->get_queue()));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZdrot_64 | FileCheck %s -check-prefix=cublasZdrot_64
// cublasZdrot_64: CUDA API:
// cublasZdrot_64-NEXT:   cublasZdrot_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasZdrot_64-NEXT:                  x /*cuDoubleComplex **/, incx /*int64_t*/,
// cublasZdrot_64-NEXT:                  y /*cuDoubleComplex **/, incy /*int64_t*/,
// cublasZdrot_64-NEXT:                  c /*const double **/, s /*const double **/);
// cublasZdrot_64-NEXT: Is migrated to:
// cublasZdrot_64-NEXT:   oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, dpct::get_value(c, handle->get_queue()), dpct::get_value(s, handle->get_queue()));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSscal_64 | FileCheck %s -check-prefix=cublasSscal_64
// cublasSscal_64: CUDA API:
// cublasSscal_64-NEXT:   cublasSscal_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasSscal_64-NEXT:                  alpha /*const float **/, x /*float **/, incx /*int64_t*/);
// cublasSscal_64-NEXT: Is migrated to:
// cublasSscal_64-NEXT:   oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDscal_64 | FileCheck %s -check-prefix=cublasDscal_64
// cublasDscal_64: CUDA API:
// cublasDscal_64-NEXT:   cublasDscal_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasDscal_64-NEXT:                  alpha /*const double **/, x /*double **/, incx /*int64_t*/);
// cublasDscal_64-NEXT: Is migrated to:
// cublasDscal_64-NEXT:   oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCscal_64 | FileCheck %s -check-prefix=cublasCscal_64
// cublasCscal_64: CUDA API:
// cublasCscal_64-NEXT:   cublasCscal_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasCscal_64-NEXT:                  alpha /*const cuComplex **/, x /*cuComplex **/,
// cublasCscal_64-NEXT:                  incx /*int64_t*/);
// cublasCscal_64-NEXT: Is migrated to:
// cublasCscal_64-NEXT:   oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsscal_64 | FileCheck %s -check-prefix=cublasCsscal_64
// cublasCsscal_64: CUDA API:
// cublasCsscal_64-NEXT:   cublasCsscal_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasCsscal_64-NEXT:                   alpha /*const float **/, x /*cuComplex **/, incx /*int64_t*/);
// cublasCsscal_64-NEXT: Is migrated to:
// cublasCsscal_64-NEXT:   oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZscal_64 | FileCheck %s -check-prefix=cublasZscal_64
// cublasZscal_64: CUDA API:
// cublasZscal_64-NEXT:   cublasZscal_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasZscal_64-NEXT:                  alpha /*const cuDoubleComplex **/, x /*cuDoubleComplex **/,
// cublasZscal_64-NEXT:                  incx /*int64_t*/);
// cublasZscal_64-NEXT: Is migrated to:
// cublasZscal_64-NEXT:   oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZdscal_64 | FileCheck %s -check-prefix=cublasZdscal_64
// cublasZdscal_64: CUDA API:
// cublasZdscal_64-NEXT:   cublasZdscal_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasZdscal_64-NEXT:                   alpha /*const double **/, x /*cuDoubleComplex **/,
// cublasZdscal_64-NEXT:                   incx /*int64_t*/);
// cublasZdscal_64-NEXT: Is migrated to:
// cublasZdscal_64-NEXT:   oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSswap_64 | FileCheck %s -check-prefix=cublasSswap_64
// cublasSswap_64: CUDA API:
// cublasSswap_64-NEXT:   cublasSswap_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*float **/,
// cublasSswap_64-NEXT:                  incx /*int64_t*/, y /*float **/, incy /*int64_t*/);
// cublasSswap_64-NEXT: Is migrated to:
// cublasSswap_64-NEXT:   oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDswap_64 | FileCheck %s -check-prefix=cublasDswap_64
// cublasDswap_64: CUDA API:
// cublasDswap_64-NEXT:   cublasDswap_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*double **/,
// cublasDswap_64-NEXT:                  incx /*int64_t*/, y /*double **/, incy /*int64_t*/);
// cublasDswap_64-NEXT: Is migrated to:
// cublasDswap_64-NEXT:   oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCswap_64 | FileCheck %s -check-prefix=cublasCswap_64
// cublasCswap_64: CUDA API:
// cublasCswap_64-NEXT:   cublasCswap_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*cuComplex **/,
// cublasCswap_64-NEXT:                  incx /*int64_t*/, y /*cuComplex **/, incy /*int64_t*/);
// cublasCswap_64-NEXT: Is migrated to:
// cublasCswap_64-NEXT:   oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZswap_64 | FileCheck %s -check-prefix=cublasZswap_64
// cublasZswap_64: CUDA API:
// cublasZswap_64-NEXT:   cublasZswap_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasZswap_64-NEXT:                  x /*cuDoubleComplex **/, incx /*int64_t*/,
// cublasZswap_64-NEXT:                  y /*cuDoubleComplex **/, incy /*int64_t*/);
// cublasZswap_64-NEXT: Is migrated to:
// cublasZswap_64-NEXT:   oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsymm_64 | FileCheck %s -check-prefix=cublasSsymm_64
// cublasSsymm_64: CUDA API:
// cublasSsymm_64-NEXT:   cublasSsymm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasSsymm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasSsymm_64-NEXT:                  alpha /*const float **/, a /*const float **/, lda /*int64_t*/,
// cublasSsymm_64-NEXT:                  b /*const float **/, ldb /*int64_t*/, beta /*const float **/,
// cublasSsymm_64-NEXT:                  c /*float **/, ldc /*int64_t*/);
// cublasSsymm_64-NEXT: Is migrated to:
// cublasSsymm_64-NEXT:   oneapi::mkl::blas::column_major::symm(handle->get_queue(), left_right, upper_lower, m, n, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsymm_64 | FileCheck %s -check-prefix=cublasDsymm_64
// cublasDsymm_64: CUDA API:
// cublasDsymm_64-NEXT:   cublasDsymm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasDsymm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasDsymm_64-NEXT:                  alpha /*const double **/, a /*const double **/,
// cublasDsymm_64-NEXT:                  lda /*int64_t*/, b /*const double **/, ldb /*int64_t*/,
// cublasDsymm_64-NEXT:                  beta /*const double **/, c /*double **/, ldc /*int64_t*/);
// cublasDsymm_64-NEXT: Is migrated to:
// cublasDsymm_64-NEXT:   oneapi::mkl::blas::column_major::symm(handle->get_queue(), left_right, upper_lower, m, n, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsymm_64 | FileCheck %s -check-prefix=cublasCsymm_64
// cublasCsymm_64: CUDA API:
// cublasCsymm_64-NEXT:   cublasCsymm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasCsymm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasCsymm_64-NEXT:                  alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCsymm_64-NEXT:                  lda /*int64_t*/, b /*const cuComplex **/, ldb /*int64_t*/,
// cublasCsymm_64-NEXT:                  beta /*const cuComplex **/, c /*cuComplex **/,
// cublasCsymm_64-NEXT:                  ldc /*int64_t*/);
// cublasCsymm_64-NEXT: Is migrated to:
// cublasCsymm_64-NEXT:   oneapi::mkl::blas::column_major::symm(handle->get_queue(), left_right, upper_lower, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsymm_64 | FileCheck %s -check-prefix=cublasZsymm_64
// cublasZsymm_64: CUDA API:
// cublasZsymm_64-NEXT:   cublasZsymm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZsymm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasZsymm_64-NEXT:                  alpha /*const cuDoubleComplex **/,
// cublasZsymm_64-NEXT:                  a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZsymm_64-NEXT:                  b /*const cuDoubleComplex **/, ldb /*int64_t*/,
// cublasZsymm_64-NEXT:                  beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZsymm_64-NEXT:                  ldc /*int64_t*/);
// cublasZsymm_64-NEXT: Is migrated to:
// cublasZsymm_64-NEXT:   oneapi::mkl::blas::column_major::symm(handle->get_queue(), left_right, upper_lower, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsyr2k_64 | FileCheck %s -check-prefix=cublasSsyr2k_64
// cublasSsyr2k_64: CUDA API:
// cublasSsyr2k_64-NEXT:   cublasSsyr2k_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsyr2k_64-NEXT:                   trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasSsyr2k_64-NEXT:                   alpha /*const float **/, a /*const float **/, lda /*int64_t*/,
// cublasSsyr2k_64-NEXT:                   b /*const float **/, ldb /*int64_t*/, beta /*const float **/,
// cublasSsyr2k_64-NEXT:                   c /*float **/, ldc /*int64_t*/);
// cublasSsyr2k_64-NEXT: Is migrated to:
// cublasSsyr2k_64-NEXT:   oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyr2k_64 | FileCheck %s -check-prefix=cublasDsyr2k_64
// cublasDsyr2k_64: CUDA API:
// cublasDsyr2k_64-NEXT:   cublasDsyr2k_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyr2k_64-NEXT:                   trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasDsyr2k_64-NEXT:                   alpha /*const double **/, a /*const double **/,
// cublasDsyr2k_64-NEXT:                   lda /*int64_t*/, b /*const double **/, ldb /*int64_t*/,
// cublasDsyr2k_64-NEXT:                   beta /*const double **/, c /*double **/, ldc /*int64_t*/);
// cublasDsyr2k_64-NEXT: Is migrated to:
// cublasDsyr2k_64-NEXT:   oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyr2k_64 | FileCheck %s -check-prefix=cublasCsyr2k_64
// cublasCsyr2k_64: CUDA API:
// cublasCsyr2k_64-NEXT:   cublasCsyr2k_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyr2k_64-NEXT:                   trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasCsyr2k_64-NEXT:                   alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCsyr2k_64-NEXT:                   lda /*int64_t*/, b /*const cuComplex **/, ldb /*int64_t*/,
// cublasCsyr2k_64-NEXT:                   beta /*const cuComplex **/, c /*cuComplex **/,
// cublasCsyr2k_64-NEXT:                   ldc /*int64_t*/);
// cublasCsyr2k_64-NEXT: Is migrated to:
// cublasCsyr2k_64-NEXT:   oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyr2k_64 | FileCheck %s -check-prefix=cublasZsyr2k_64
// cublasZsyr2k_64: CUDA API:
// cublasZsyr2k_64-NEXT:   cublasZsyr2k_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyr2k_64-NEXT:                   trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasZsyr2k_64-NEXT:                   alpha /*const cuDoubleComplex **/,
// cublasZsyr2k_64-NEXT:                   a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZsyr2k_64-NEXT:                   b /*const cuDoubleComplex **/, ldb /*int64_t*/,
// cublasZsyr2k_64-NEXT:                   beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZsyr2k_64-NEXT:                   ldc /*int64_t*/);
// cublasZsyr2k_64-NEXT: Is migrated to:
// cublasZsyr2k_64-NEXT:   oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc);
