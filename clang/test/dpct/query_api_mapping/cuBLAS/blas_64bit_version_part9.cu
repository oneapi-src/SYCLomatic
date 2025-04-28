// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyr_64 | FileCheck %s -check-prefix=cublasZsyr_64
// cublasZsyr_64: CUDA API:
// cublasZsyr_64-NEXT:   cublasZsyr_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyr_64-NEXT:                 n /*int64_t*/, alpha /*const cuDoubleComplex **/,
// cublasZsyr_64-NEXT:                 x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZsyr_64-NEXT:                 a /*cuDoubleComplex **/, lda /*int64_t*/);
// cublasZsyr_64-NEXT: Is migrated to:
// cublasZsyr_64-NEXT:   oneapi::mkl::blas::column_major::syr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyrk3mEx_64 | FileCheck %s -check-prefix=cublasCsyrk3mEx_64
// cublasCsyrk3mEx_64: CUDA API:
// cublasCsyrk3mEx_64-NEXT:   cublasCsyrk3mEx_64(handle /*cublasHandle_t*/, uplo /*cublasFillMode_t*/,
// cublasCsyrk3mEx_64-NEXT:                      trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasCsyrk3mEx_64-NEXT:                      alpha /*const cuComplex **/, a /*const void **/,
// cublasCsyrk3mEx_64-NEXT:                      a_type /*cudaDataType*/, lda /*int64_t*/,
// cublasCsyrk3mEx_64-NEXT:                      beta /*const cuComplex **/, c /*void **/,
// cublasCsyrk3mEx_64-NEXT:                      c_type /*cudaDataType*/, ldc /*int64_t*/);
// cublasCsyrk3mEx_64-NEXT: Is migrated to:
// cublasCsyrk3mEx_64-NEXT:   dpct::blas::syherk<false>(handle, uplo, trans, n, k, alpha, a, a_type, lda, beta, c, c_type, ldc, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyrkEx_64 | FileCheck %s -check-prefix=cublasCsyrkEx_64
// cublasCsyrkEx_64: CUDA API:
// cublasCsyrkEx_64-NEXT:   cublasCsyrkEx_64(handle /*cublasHandle_t*/, uplo /*cublasFillMode_t*/,
// cublasCsyrkEx_64-NEXT:                    trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasCsyrkEx_64-NEXT:                    alpha /*const cuComplex **/, a /*const void **/,
// cublasCsyrkEx_64-NEXT:                    a_type /*cudaDataType*/, lda /*int64_t*/,
// cublasCsyrkEx_64-NEXT:                    beta /*const cuComplex **/, c /*void **/,
// cublasCsyrkEx_64-NEXT:                    c_type /*cudaDataType*/, ldc /*int64_t*/);
// cublasCsyrkEx_64-NEXT: Is migrated to:
// cublasCsyrkEx_64-NEXT:   dpct::blas::syherk<false>(handle, uplo, trans, n, k, alpha, a, a_type, lda, beta, c, c_type, ldc, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyrkx_64 | FileCheck %s -check-prefix=cublasCsyrkx_64
// cublasCsyrkx_64: CUDA API:
// cublasCsyrkx_64-NEXT:   cublasCsyrkx_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyrkx_64-NEXT:                   trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasCsyrkx_64-NEXT:                   alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCsyrkx_64-NEXT:                   lda /*int64_t*/, b /*const cuComplex **/, ldb /*int64_t*/,
// cublasCsyrkx_64-NEXT:                   beta /*const cuComplex **/, c /*cuComplex **/,
// cublasCsyrkx_64-NEXT:                   ldc /*int64_t*/);
// cublasCsyrkx_64-NEXT: Is migrated to:
// cublasCsyrkx_64-NEXT:   dpct::blas::syrk(handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyrkx_64 | FileCheck %s -check-prefix=cublasDsyrkx_64
// cublasDsyrkx_64: CUDA API:
// cublasDsyrkx_64-NEXT:   cublasDsyrkx_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyrkx_64-NEXT:                   trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasDsyrkx_64-NEXT:                   alpha /*const double **/, a /*const double **/,
// cublasDsyrkx_64-NEXT:                   lda /*int64_t*/, b /*const double **/, ldb /*int64_t*/,
// cublasDsyrkx_64-NEXT:                   beta /*const double **/, c /*double **/, ldc /*int64_t*/);
// cublasDsyrkx_64-NEXT: Is migrated to:
// cublasDsyrkx_64-NEXT:   dpct::blas::syrk(handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsyrkx_64 | FileCheck %s -check-prefix=cublasSsyrkx_64
// cublasSsyrkx_64: CUDA API:
// cublasSsyrkx_64-NEXT:   cublasSsyrkx_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsyrkx_64-NEXT:                   trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasSsyrkx_64-NEXT:                   alpha /*const float **/, a /*const float **/, lda /*int64_t*/,
// cublasSsyrkx_64-NEXT:                   b /*const float **/, ldb /*int64_t*/, beta /*const float **/,
// cublasSsyrkx_64-NEXT:                   c /*float **/, ldc /*int64_t*/);
// cublasSsyrkx_64-NEXT: Is migrated to:
// cublasSsyrkx_64-NEXT:   dpct::blas::syrk(handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyrkx_64 | FileCheck %s -check-prefix=cublasZsyrkx_64
// cublasZsyrkx_64: CUDA API:
// cublasZsyrkx_64-NEXT:   cublasZsyrkx_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyrkx_64-NEXT:                   trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasZsyrkx_64-NEXT:                   alpha /*const cuDoubleComplex **/,
// cublasZsyrkx_64-NEXT:                   a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZsyrkx_64-NEXT:                   b /*const cuDoubleComplex **/, ldb /*int64_t*/,
// cublasZsyrkx_64-NEXT:                   beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZsyrkx_64-NEXT:                   ldc /*int64_t*/);
// cublasZsyrkx_64-NEXT: Is migrated to:
// cublasZsyrkx_64-NEXT:   dpct::blas::syrk(handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtbsv_64 | FileCheck %s -check-prefix=cublasCtbsv_64
// cublasCtbsv_64: CUDA API:
// cublasCtbsv_64-NEXT:   cublasCtbsv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtbsv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtbsv_64-NEXT:                  n /*int64_t*/, k /*int64_t*/, a /*const cuComplex **/,
// cublasCtbsv_64-NEXT:                  lda /*int64_t*/, x /*cuComplex **/, incx /*int64_t*/);
// cublasCtbsv_64-NEXT: Is migrated to:
// cublasCtbsv_64-NEXT:   oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtbsv_64 | FileCheck %s -check-prefix=cublasDtbsv_64
// cublasDtbsv_64: CUDA API:
// cublasDtbsv_64-NEXT:   cublasDtbsv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtbsv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtbsv_64-NEXT:                  n /*int64_t*/, k /*int64_t*/, a /*const double **/,
// cublasDtbsv_64-NEXT:                  lda /*int64_t*/, x /*double **/, incx /*int64_t*/);
// cublasDtbsv_64-NEXT: Is migrated to:
// cublasDtbsv_64-NEXT:   oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStbsv_64 | FileCheck %s -check-prefix=cublasStbsv_64
// cublasStbsv_64: CUDA API:
// cublasStbsv_64-NEXT:   cublasStbsv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStbsv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStbsv_64-NEXT:                  n /*int64_t*/, k /*int64_t*/, a /*const float **/,
// cublasStbsv_64-NEXT:                  lda /*int64_t*/, x /*float **/, incx /*int64_t*/);
// cublasStbsv_64-NEXT: Is migrated to:
// cublasStbsv_64-NEXT:   oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtbsv_64 | FileCheck %s -check-prefix=cublasZtbsv_64
// cublasZtbsv_64: CUDA API:
// cublasZtbsv_64-NEXT:   cublasZtbsv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtbsv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtbsv_64-NEXT:                  n /*int64_t*/, k /*int64_t*/, a /*const cuDoubleComplex **/,
// cublasZtbsv_64-NEXT:                  lda /*int64_t*/, x /*cuDoubleComplex **/, incx /*int64_t*/);
// cublasZtbsv_64-NEXT: Is migrated to:
// cublasZtbsv_64-NEXT:   oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtrmm_64 | FileCheck %s -check-prefix=cublasCtrmm_64
// cublasCtrmm_64: CUDA API:
// cublasCtrmm_64-NEXT:   cublasCtrmm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasCtrmm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasCtrmm_64-NEXT:                  unit_diag /*cublasDiagType_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasCtrmm_64-NEXT:                  alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCtrmm_64-NEXT:                  lda /*int64_t*/, b /*const cuComplex **/, ldb /*int64_t*/,
// cublasCtrmm_64-NEXT:                  c /*cuComplex **/, ldc /*int64_t*/);
// cublasCtrmm_64-NEXT: Is migrated to:
// cublasCtrmm_64-NEXT:   dpct::blas::trmm(handle, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtrmm_64 | FileCheck %s -check-prefix=cublasDtrmm_64
// cublasDtrmm_64: CUDA API:
// cublasDtrmm_64-NEXT:   cublasDtrmm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasDtrmm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasDtrmm_64-NEXT:                  unit_diag /*cublasDiagType_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasDtrmm_64-NEXT:                  alpha /*const double **/, a /*const double **/,
// cublasDtrmm_64-NEXT:                  lda /*int64_t*/, b /*const double **/, ldb /*int64_t*/,
// cublasDtrmm_64-NEXT:                  c /*double **/, ldc /*int64_t*/);
// cublasDtrmm_64-NEXT: Is migrated to:
// cublasDtrmm_64-NEXT:   dpct::blas::trmm(handle, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStrmm_64 | FileCheck %s -check-prefix=cublasStrmm_64
// cublasStrmm_64: CUDA API:
// cublasStrmm_64-NEXT:   cublasStrmm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasStrmm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasStrmm_64-NEXT:                  unit_diag /*cublasDiagType_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasStrmm_64-NEXT:                  alpha /*const float **/, a /*const float **/, lda /*int64_t*/,
// cublasStrmm_64-NEXT:                  b /*const float **/, ldb /*int64_t*/, c /*float **/,
// cublasStrmm_64-NEXT:                  ldc /*int64_t*/);
// cublasStrmm_64-NEXT: Is migrated to:
// cublasStrmm_64-NEXT:   dpct::blas::trmm(handle, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtrmm_64 | FileCheck %s -check-prefix=cublasZtrmm_64
// cublasZtrmm_64: CUDA API:
// cublasZtrmm_64-NEXT:   cublasZtrmm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZtrmm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasZtrmm_64-NEXT:                  unit_diag /*cublasDiagType_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasZtrmm_64-NEXT:                  alpha /*const cuDoubleComplex **/,
// cublasZtrmm_64-NEXT:                  a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZtrmm_64-NEXT:                  b /*const cuDoubleComplex **/, ldb /*int64_t*/,
// cublasZtrmm_64-NEXT:                  c /*cuDoubleComplex **/, ldc /*int64_t*/);
// cublasZtrmm_64-NEXT: Is migrated to:
// cublasZtrmm_64-NEXT:   dpct::blas::trmm(handle, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDger_64 | FileCheck %s -check-prefix=cublasDger_64
// cublasDger_64: CUDA API:
// cublasDger_64-NEXT:   cublasDger_64(handle /*cublasHandle_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasDger_64-NEXT:                 alpha /*const double **/, x /*const double **/,
// cublasDger_64-NEXT:                 incx /*int64_t*/, y /*const double **/, incy /*int64_t*/,
// cublasDger_64-NEXT:                 a /*double **/, lda /*int64_t*/);
// cublasDger_64-NEXT: Is migrated to:
// cublasDger_64-NEXT:   oneapi::mkl::blas::column_major::ger(handle->get_queue(), m, n, dpct::get_value(alpha, handle->get_queue()), x, incx, y, incy, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSger_64 | FileCheck %s -check-prefix=cublasSger_64
// cublasSger_64: CUDA API:
// cublasSger_64-NEXT:   cublasSger_64(handle /*cublasHandle_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasSger_64-NEXT:                 alpha /*const float **/, x /*const float **/, incx /*int64_t*/,
// cublasSger_64-NEXT:                 y /*const float **/, incy /*int64_t*/, a /*float **/,
// cublasSger_64-NEXT:                 lda /*int64_t*/);
// cublasSger_64-NEXT: Is migrated to:
// cublasSger_64-NEXT:   oneapi::mkl::blas::column_major::ger(handle->get_queue(), m, n, dpct::get_value(alpha, handle->get_queue()), x, incx, y, incy, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDotcEx_64 | FileCheck %s -check-prefix=cublasDotcEx_64
// cublasDotcEx_64: CUDA API:
// cublasDotcEx_64-NEXT:   cublasDotcEx_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const void **/,
// cublasDotcEx_64-NEXT:                   xtype /*cudaDataType*/, incx /*int64_t*/, y /*const void **/,
// cublasDotcEx_64-NEXT:                   ytype /*cudaDataType*/, incy /*int64_t*/, res /*void **/,
// cublasDotcEx_64-NEXT:                   restype /*cudaDataType*/, computetype /*cudaDataType*/);
// cublasDotcEx_64-NEXT: Is migrated to:
// cublasDotcEx_64-NEXT:   dpct::blas::dotc(handle, n, x, xtype, incx, y, ytype, incy, res, restype);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDotEx_64 | FileCheck %s -check-prefix=cublasDotEx_64
// cublasDotEx_64: CUDA API:
// cublasDotEx_64-NEXT:   cublasDotEx_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const void **/,
// cublasDotEx_64-NEXT:                  xtype /*cudaDataType*/, incx /*int64_t*/, y /*const void **/,
// cublasDotEx_64-NEXT:                  ytype /*cudaDataType*/, incy /*int64_t*/, res /*void **/,
// cublasDotEx_64-NEXT:                  restype /*cudaDataType*/, computetype /*cudaDataType*/);
// cublasDotEx_64-NEXT: Is migrated to:
// cublasDotEx_64-NEXT:   dpct::blas::dot(handle, n, x, xtype, incx, y, ytype, incy, res, restype);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsbmv_64 | FileCheck %s -check-prefix=cublasDsbmv_64
// cublasDsbmv_64: CUDA API:
// cublasDsbmv_64-NEXT:   cublasDsbmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsbmv_64-NEXT:                  n /*int64_t*/, k /*int64_t*/, alpha /*const double **/,
// cublasDsbmv_64-NEXT:                  a /*const double **/, lda /*int64_t*/, x /*const double **/,
// cublasDsbmv_64-NEXT:                  incx /*int64_t*/, beta /*const double **/, y /*double **/,
// cublasDsbmv_64-NEXT:                  incy /*int64_t*/);
// cublasDsbmv_64-NEXT: Is migrated to:
// cublasDsbmv_64-NEXT:   oneapi::mkl::blas::column_major::sbmv(handle->get_queue(), upper_lower, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);
