// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemm3mEx_64 | FileCheck %s -check-prefix=cublasCgemm3mEx_64
// cublasCgemm3mEx_64: CUDA API:
// cublasCgemm3mEx_64-NEXT:   cublasCgemm3mEx_64(
// cublasCgemm3mEx_64-NEXT:       handle /*cublasHandle_t*/, trans_a /*cublasOperation_t*/,
// cublasCgemm3mEx_64-NEXT:       trans_b /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasCgemm3mEx_64-NEXT:       k /*int64_t*/, alpha /*const cuComplex **/, a /*const void **/,
// cublasCgemm3mEx_64-NEXT:       a_type /*cudaDataType*/, lda /*int64_t*/, b /*const void **/,
// cublasCgemm3mEx_64-NEXT:       b_type /*cudaDataType*/, ldb /*int64_t*/, beta /*const cuComplex **/,
// cublasCgemm3mEx_64-NEXT:       c /*void **/, c_type /*cudaDataType*/, ldc /*int64_t*/);
// cublasCgemm3mEx_64-NEXT: Is migrated to:
// cublasCgemm3mEx_64-NEXT:   dpct::blas::gemm(handle, trans_a, trans_b, m, n, k, alpha, a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemmEx_64 | FileCheck %s -check-prefix=cublasCgemmEx_64
// cublasCgemmEx_64: CUDA API:
// cublasCgemmEx_64-NEXT:   cublasCgemmEx_64(handle /*cublasHandle_t*/, trans_a /*cublasOperation_t*/,
// cublasCgemmEx_64-NEXT:                    trans_b /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasCgemmEx_64-NEXT:                    k /*int64_t*/, alpha /*const cuComplex **/,
// cublasCgemmEx_64-NEXT:                    a /*const void **/, a_type /*cudaDataType*/, lda /*int64_t*/,
// cublasCgemmEx_64-NEXT:                    b /*const void **/, b_type /*cudaDataType*/, ldb /*int64_t*/,
// cublasCgemmEx_64-NEXT:                    beta /*const cuComplex **/, c /*void **/,
// cublasCgemmEx_64-NEXT:                    c_type /*cudaDataType*/, ldc /*int64_t*/);
// cublasCgemmEx_64-NEXT: Is migrated to:
// cublasCgemmEx_64-NEXT:   dpct::blas::gemm(handle, trans_a, trans_b, m, n, k, alpha, a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgerc_64 | FileCheck %s -check-prefix=cublasCgerc_64
// cublasCgerc_64: CUDA API:
// cublasCgerc_64-NEXT:   cublasCgerc_64(handle /*cublasHandle_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasCgerc_64-NEXT:                  alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasCgerc_64-NEXT:                  incx /*int64_t*/, y /*const cuComplex **/, incy /*int64_t*/,
// cublasCgerc_64-NEXT:                  a /*cuComplex **/, lda /*int64_t*/);
// cublasCgerc_64-NEXT: Is migrated to:
// cublasCgerc_64-NEXT:   oneapi::mkl::blas::column_major::gerc(handle->get_queue(), m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgerc_64 | FileCheck %s -check-prefix=cublasZgerc_64
// cublasZgerc_64: CUDA API:
// cublasZgerc_64-NEXT:   cublasZgerc_64(handle /*cublasHandle_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasZgerc_64-NEXT:                  alpha /*const cuDoubleComplex **/,
// cublasZgerc_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZgerc_64-NEXT:                  y /*const cuDoubleComplex **/, incy /*int64_t*/,
// cublasZgerc_64-NEXT:                  a /*cuDoubleComplex **/, lda /*int64_t*/);
// cublasZgerc_64-NEXT: Is migrated to:
// cublasZgerc_64-NEXT:   oneapi::mkl::blas::column_major::gerc(handle->get_queue(), m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgeru_64 | FileCheck %s -check-prefix=cublasCgeru_64
// cublasCgeru_64: CUDA API:
// cublasCgeru_64-NEXT:   cublasCgeru_64(handle /*cublasHandle_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasCgeru_64-NEXT:                  alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasCgeru_64-NEXT:                  incx /*int64_t*/, y /*const cuComplex **/, incy /*int64_t*/,
// cublasCgeru_64-NEXT:                  a /*cuComplex **/, lda /*int64_t*/);
// cublasCgeru_64-NEXT: Is migrated to:
// cublasCgeru_64-NEXT:   oneapi::mkl::blas::column_major::geru(handle->get_queue(), m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgeru_64 | FileCheck %s -check-prefix=cublasZgeru_64
// cublasZgeru_64: CUDA API:
// cublasZgeru_64-NEXT:   cublasZgeru_64(handle /*cublasHandle_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasZgeru_64-NEXT:                  alpha /*const cuDoubleComplex **/,
// cublasZgeru_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZgeru_64-NEXT:                  y /*const cuDoubleComplex **/, incy /*int64_t*/,
// cublasZgeru_64-NEXT:                  a /*cuDoubleComplex **/, lda /*int64_t*/);
// cublasZgeru_64-NEXT: Is migrated to:
// cublasZgeru_64-NEXT:   oneapi::mkl::blas::column_major::geru(handle->get_queue(), m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChbmv_64 | FileCheck %s -check-prefix=cublasChbmv_64
// cublasChbmv_64: CUDA API:
// cublasChbmv_64-NEXT:   cublasChbmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChbmv_64-NEXT:                  n /*int64_t*/, k /*int64_t*/, alpha /*const cuComplex **/,
// cublasChbmv_64-NEXT:                  a /*const cuComplex **/, lda /*int64_t*/,
// cublasChbmv_64-NEXT:                  x /*const cuComplex **/, incx /*int64_t*/,
// cublasChbmv_64-NEXT:                  beta /*const cuComplex **/, y /*cuComplex **/,
// cublasChbmv_64-NEXT:                  incy /*int64_t*/);
// cublasChbmv_64-NEXT: Is migrated to:
// cublasChbmv_64-NEXT:   oneapi::mkl::blas::column_major::hbmv(handle->get_queue(), upper_lower, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhbmv_64 | FileCheck %s -check-prefix=cublasZhbmv_64
// cublasZhbmv_64: CUDA API:
// cublasZhbmv_64-NEXT:   cublasZhbmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZhbmv_64-NEXT:                  n /*int64_t*/, k /*int64_t*/,
// cublasZhbmv_64-NEXT:                  alpha /*const cuDoubleComplex **/,
// cublasZhbmv_64-NEXT:                  a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZhbmv_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZhbmv_64-NEXT:                  beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/,
// cublasZhbmv_64-NEXT:                  incy /*int64_t*/);
// cublasZhbmv_64-NEXT: Is migrated to:
// cublasZhbmv_64-NEXT:   oneapi::mkl::blas::column_major::hbmv(handle->get_queue(), upper_lower, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChemv_64 | FileCheck %s -check-prefix=cublasChemv_64
// cublasChemv_64: CUDA API:
// cublasChemv_64-NEXT:   cublasChemv_64(
// cublasChemv_64-NEXT:       handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChemv_64-NEXT:       n /*int64_t*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasChemv_64-NEXT:       lda /*int64_t*/, x /*const cuComplex **/, incx /*int64_t*/,
// cublasChemv_64-NEXT:       beta /*const cuComplex **/, y /*cuComplex **/, incy /*int64_t*/);
// cublasChemv_64-NEXT: Is migrated to:
// cublasChemv_64-NEXT:   oneapi::mkl::blas::column_major::hemv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhemv_64 | FileCheck %s -check-prefix=cublasZhemv_64
// cublasZhemv_64: CUDA API:
// cublasZhemv_64-NEXT:   cublasZhemv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZhemv_64-NEXT:                  n /*int64_t*/, alpha /*const cuDoubleComplex **/,
// cublasZhemv_64-NEXT:                  a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZhemv_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZhemv_64-NEXT:                  beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/,
// cublasZhemv_64-NEXT:                  incy /*int64_t*/);
// cublasZhemv_64-NEXT: Is migrated to:
// cublasZhemv_64-NEXT:   oneapi::mkl::blas::column_major::hemv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCher2_64 | FileCheck %s -check-prefix=cublasCher2_64
// cublasCher2_64: CUDA API:
// cublasCher2_64-NEXT:   cublasCher2_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCher2_64-NEXT:                  n /*int64_t*/, alpha /*const cuComplex **/,
// cublasCher2_64-NEXT:                  x /*const cuComplex **/, incx /*int64_t*/,
// cublasCher2_64-NEXT:                  y /*const cuComplex **/, incy /*int64_t*/, a /*cuComplex **/,
// cublasCher2_64-NEXT:                  lda /*int64_t*/);
// cublasCher2_64-NEXT: Is migrated to:
// cublasCher2_64-NEXT:   oneapi::mkl::blas::column_major::her2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZher2_64 | FileCheck %s -check-prefix=cublasZher2_64
// cublasZher2_64: CUDA API:
// cublasZher2_64-NEXT:   cublasZher2_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZher2_64-NEXT:                  n /*int64_t*/, alpha /*const cuDoubleComplex **/,
// cublasZher2_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZher2_64-NEXT:                  y /*const cuDoubleComplex **/, incy /*int64_t*/,
// cublasZher2_64-NEXT:                  a /*cuDoubleComplex **/, lda /*int64_t*/);
// cublasZher2_64-NEXT: Is migrated to:
// cublasZher2_64-NEXT:   oneapi::mkl::blas::column_major::her2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCher_64 | FileCheck %s -check-prefix=cublasCher_64
// cublasCher_64: CUDA API:
// cublasCher_64-NEXT:   cublasCher_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCher_64-NEXT:                 n /*int64_t*/, alpha /*const float **/, x /*const cuComplex **/,
// cublasCher_64-NEXT:                 incx /*int64_t*/, a /*cuComplex **/, lda /*int64_t*/);
// cublasCher_64-NEXT: Is migrated to:
// cublasCher_64-NEXT:   oneapi::mkl::blas::column_major::her(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZher_64 | FileCheck %s -check-prefix=cublasZher_64
// cublasZher_64: CUDA API:
// cublasZher_64-NEXT:   cublasZher_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZher_64-NEXT:                 n /*int64_t*/, alpha /*const double **/,
// cublasZher_64-NEXT:                 x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZher_64-NEXT:                 a /*cuDoubleComplex **/, lda /*int64_t*/);
// cublasZher_64-NEXT: Is migrated to:
// cublasZher_64-NEXT:   oneapi::mkl::blas::column_major::her(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemm3m_64 | FileCheck %s -check-prefix=cublasCgemm3m_64
// cublasCgemm3m_64: CUDA API:
// cublasCgemm3m_64-NEXT:   cublasCgemm3m_64(
// cublasCgemm3m_64-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasCgemm3m_64-NEXT:       transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasCgemm3m_64-NEXT:       alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int64_t*/,
// cublasCgemm3m_64-NEXT:       b /*const cuComplex **/, ldb /*int64_t*/, beta /*const cuComplex **/,
// cublasCgemm3m_64-NEXT:       c /*cuComplex **/, ldc /*int64_t*/);
// cublasCgemm3m_64-NEXT: Is migrated to:
// cublasCgemm3m_64-NEXT:   oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)c, ldc, oneapi::mkl::blas::compute_mode::complex_3m);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgemm3m_64 | FileCheck %s -check-prefix=cublasZgemm3m_64
// cublasZgemm3m_64: CUDA API:
// cublasZgemm3m_64-NEXT:   cublasZgemm3m_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasZgemm3m_64-NEXT:                    transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasZgemm3m_64-NEXT:                    k /*int64_t*/, alpha /*const cuDoubleComplex **/,
// cublasZgemm3m_64-NEXT:                    a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZgemm3m_64-NEXT:                    b /*const cuDoubleComplex **/, ldb /*int64_t*/,
// cublasZgemm3m_64-NEXT:                    beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZgemm3m_64-NEXT:                    ldc /*int64_t*/);
// cublasZgemm3m_64-NEXT: Is migrated to:
// cublasZgemm3m_64-NEXT:   oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc, oneapi::mkl::blas::compute_mode::complex_3m);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasAsumEx_64 | FileCheck %s -check-prefix=cublasAsumEx_64
// cublasAsumEx_64: CUDA API:
// cublasAsumEx_64-NEXT:   cublasAsumEx_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const void **/,
// cublasAsumEx_64-NEXT:                   x_type /*cudaDataType*/, incx /*int64_t*/, result /*void **/,
// cublasAsumEx_64-NEXT:                   result_type /*cudaDataType*/,
// cublasAsumEx_64-NEXT:                   execution_type /*cudaDataType*/);
// cublasAsumEx_64-NEXT: Is migrated to:
// cublasAsumEx_64-NEXT:   dpct::blas::asum(handle, n, x, x_type, incx, result, result_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasAxpyEx_64 | FileCheck %s -check-prefix=cublasAxpyEx_64
// cublasAxpyEx_64: CUDA API:
// cublasAxpyEx_64-NEXT:   cublasAxpyEx_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasAxpyEx_64-NEXT:                   alpha /*const void **/, alphatype /*cudaDataType*/,
// cublasAxpyEx_64-NEXT:                   x /*const void **/, xtype /*cudaDataType*/, incx /*int64_t*/,
// cublasAxpyEx_64-NEXT:                   y /*void **/, ytype /*cudaDataType*/, incy /*int64_t*/,
// cublasAxpyEx_64-NEXT:                   computetype /*cudaDataType*/);
// cublasAxpyEx_64-NEXT: Is migrated to:
// cublasAxpyEx_64-NEXT:   dpct::blas::axpy(handle, n, alpha, alphatype, x, xtype, incx, y, ytype, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCherk3mEx_64 | FileCheck %s -check-prefix=cublasCherk3mEx_64
// cublasCherk3mEx_64: CUDA API:
// cublasCherk3mEx_64-NEXT:   cublasCherk3mEx_64(handle /*cublasHandle_t*/, uplo /*cublasFillMode_t*/,
// cublasCherk3mEx_64-NEXT:                      trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasCherk3mEx_64-NEXT:                      alpha /*const float **/, a /*const void **/,
// cublasCherk3mEx_64-NEXT:                      a_type /*cudaDataType*/, lda /*int64_t*/,
// cublasCherk3mEx_64-NEXT:                      beta /*const float **/, c /*void **/,
// cublasCherk3mEx_64-NEXT:                      c_type /*cudaDataType*/, ldc /*int64_t*/);
// cublasCherk3mEx_64-NEXT: Is migrated to:
// cublasCherk3mEx_64-NEXT:   dpct::blas::syherk<true>(handle, uplo, trans, n, k, alpha, a, a_type, lda, beta, c, c_type, ldc, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCherkEx_64 | FileCheck %s -check-prefix=cublasCherkEx_64
// cublasCherkEx_64: CUDA API:
// cublasCherkEx_64-NEXT:   cublasCherkEx_64(handle /*cublasHandle_t*/, uplo /*cublasFillMode_t*/,
// cublasCherkEx_64-NEXT:                    trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasCherkEx_64-NEXT:                    alpha /*const float **/, a /*const void **/,
// cublasCherkEx_64-NEXT:                    a_type /*cudaDataType*/, lda /*int64_t*/,
// cublasCherkEx_64-NEXT:                    beta /*const float **/, c /*void **/,
// cublasCherkEx_64-NEXT:                    c_type /*cudaDataType*/, ldc /*int64_t*/);
// cublasCherkEx_64-NEXT: Is migrated to:
// cublasCherkEx_64-NEXT:   dpct::blas::syherk<true>(handle, uplo, trans, n, k, alpha, a, a_type, lda, beta, c, c_type, ldc, dpct::library_data_t::complex_float);
