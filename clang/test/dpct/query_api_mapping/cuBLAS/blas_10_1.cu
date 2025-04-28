// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasRotEx | FileCheck %s -check-prefix=cublasRotEx
// cublasRotEx: CUDA API:
// cublasRotEx-NEXT:   cublasRotEx(handle /*cublasHandle_t*/, n /*int*/, x /*void **/,
// cublasRotEx-NEXT:               xtype /*cudaDataType*/, incx /*int*/, y /*void **/,
// cublasRotEx-NEXT:               ytype /*cudaDataType*/, incy /*int*/, c /*const void **/,
// cublasRotEx-NEXT:               s /*const void **/, cstype /*cudaDataType*/,
// cublasRotEx-NEXT:               computetype /*cudaDataType*/);
// cublasRotEx-NEXT: Is migrated to:
// cublasRotEx-NEXT:   dpct::blas::rot(handle, n, x, xtype, incx, y, ytype, incy, c, s, cstype);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasAsumEx | FileCheck %s -check-prefix=cublasAsumEx
// cublasAsumEx: CUDA API:
// cublasAsumEx-NEXT:   cublasAsumEx(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
// cublasAsumEx-NEXT:                x_type /*cudaDataType*/, incx /*int*/, result /*void **/,
// cublasAsumEx-NEXT:                result_type /*cudaDataType*/, execution_type /*cudaDataType*/);
// cublasAsumEx-NEXT: Is migrated to:
// cublasAsumEx-NEXT:   dpct::blas::asum(handle, n, x, x_type, incx, result, result_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgelsBatched | FileCheck %s -check-prefix=cublasCgelsBatched
// cublasCgelsBatched: CUDA API:
// cublasCgelsBatched-NEXT:   cublasCgelsBatched(
// cublasCgelsBatched-NEXT:       handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasCgelsBatched-NEXT:       n /*int*/, nrhs /*int*/, a_array /*cuComplex *const []*/, lda /*int*/,
// cublasCgelsBatched-NEXT:       c_array /*cuComplex *const []*/, ldc /*int*/, info /*int **/,
// cublasCgelsBatched-NEXT:       dev_info_array /*int **/, batch_size /*int*/);
// cublasCgelsBatched-NEXT: Is migrated to:
// cublasCgelsBatched-NEXT:   dpct::blas::gels_batch_wrapper(handle, trans, m, n, nrhs, a_array, lda, c_array, ldc, info, dev_info_array, batch_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgelsBatched | FileCheck %s -check-prefix=cublasDgelsBatched
// cublasDgelsBatched: CUDA API:
// cublasDgelsBatched-NEXT:   cublasDgelsBatched(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
// cublasDgelsBatched-NEXT:                      m /*int*/, n /*int*/, nrhs /*int*/,
// cublasDgelsBatched-NEXT:                      a_array /*double *const []*/, lda /*int*/,
// cublasDgelsBatched-NEXT:                      c_array /*double *const []*/, ldc /*int*/, info /*int **/,
// cublasDgelsBatched-NEXT:                      dev_info_array /*int **/, batch_size /*int*/);
// cublasDgelsBatched-NEXT: Is migrated to:
// cublasDgelsBatched-NEXT:   dpct::blas::gels_batch_wrapper(handle, trans, m, n, nrhs, a_array, lda, c_array, ldc, info, dev_info_array, batch_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgelsBatched | FileCheck %s -check-prefix=cublasSgelsBatched
// cublasSgelsBatched: CUDA API:
// cublasSgelsBatched-NEXT:   cublasSgelsBatched(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
// cublasSgelsBatched-NEXT:                      m /*int*/, n /*int*/, nrhs /*int*/,
// cublasSgelsBatched-NEXT:                      a_array /*float *const []*/, lda /*int*/,
// cublasSgelsBatched-NEXT:                      c_array /*float *const []*/, ldc /*int*/, info /*int **/,
// cublasSgelsBatched-NEXT:                      dev_info_array /*int **/, batch_size /*int*/);
// cublasSgelsBatched-NEXT: Is migrated to:
// cublasSgelsBatched-NEXT:   dpct::blas::gels_batch_wrapper(handle, trans, m, n, nrhs, a_array, lda, c_array, ldc, info, dev_info_array, batch_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgelsBatched | FileCheck %s -check-prefix=cublasZgelsBatched
// cublasZgelsBatched: CUDA API:
// cublasZgelsBatched-NEXT:   cublasZgelsBatched(
// cublasZgelsBatched-NEXT:       handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasZgelsBatched-NEXT:       n /*int*/, nrhs /*int*/, a_array /*cuDoubleComplex *const []*/,
// cublasZgelsBatched-NEXT:       lda /*int*/, c_array /*cuDoubleComplex *const []*/, ldc /*int*/,
// cublasZgelsBatched-NEXT:       info /*int **/, dev_info_array /*int **/, batch_size /*int*/);
// cublasZgelsBatched-NEXT: Is migrated to:
// cublasZgelsBatched-NEXT:   dpct::blas::gels_batch_wrapper(handle, trans, m, n, nrhs, a_array, lda, c_array, ldc, info, dev_info_array, batch_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemm3mEx | FileCheck %s -check-prefix=cublasCgemm3mEx
// cublasCgemm3mEx: CUDA API:
// cublasCgemm3mEx-NEXT:   cublasCgemm3mEx(handle /*cublasHandle_t*/, trans_a /*cublasOperation_t*/,
// cublasCgemm3mEx-NEXT:                   trans_b /*cublasOperation_t*/, m /*int*/, n /*int*/,
// cublasCgemm3mEx-NEXT:                   k /*int*/, alpha /*const cuComplex **/, a /*const void **/,
// cublasCgemm3mEx-NEXT:                   a_type /*cudaDataType*/, lda /*int*/, b /*const void **/,
// cublasCgemm3mEx-NEXT:                   b_type /*cudaDataType*/, ldb /*int*/,
// cublasCgemm3mEx-NEXT:                   beta /*const cuComplex **/, c /*void **/,
// cublasCgemm3mEx-NEXT:                   c_type /*cudaDataType*/, ldc /*int*/);
// cublasCgemm3mEx-NEXT: Is migrated to:
// cublasCgemm3mEx-NEXT:   dpct::blas::gemm(handle, trans_a, trans_b, m, n, k, alpha, a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCherk3mEx | FileCheck %s -check-prefix=cublasCherk3mEx
// cublasCherk3mEx: CUDA API:
// cublasCherk3mEx-NEXT:   cublasCherk3mEx(handle /*cublasHandle_t*/, uplo /*cublasFillMode_t*/,
// cublasCherk3mEx-NEXT:                   trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCherk3mEx-NEXT:                   alpha /*const float **/, a /*const void **/,
// cublasCherk3mEx-NEXT:                   a_type /*cudaDataType*/, lda /*int*/, beta /*const float **/,
// cublasCherk3mEx-NEXT:                   c /*void **/, c_type /*cudaDataType*/, ldc /*int*/);
// cublasCherk3mEx-NEXT: Is migrated to:
// cublasCherk3mEx-NEXT:   dpct::blas::syherk<true>(handle, uplo, trans, n, k, alpha, a, a_type, lda, beta, c, c_type, ldc, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCherkEx | FileCheck %s -check-prefix=cublasCherkEx
// cublasCherkEx: CUDA API:
// cublasCherkEx-NEXT:   cublasCherkEx(handle /*cublasHandle_t*/, uplo /*cublasFillMode_t*/,
// cublasCherkEx-NEXT:                 trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCherkEx-NEXT:                 alpha /*const float **/, a /*const void **/,
// cublasCherkEx-NEXT:                 a_type /*cudaDataType*/, lda /*int*/, beta /*const float **/,
// cublasCherkEx-NEXT:                 c /*void **/, c_type /*cudaDataType*/, ldc /*int*/);
// cublasCherkEx-NEXT: Is migrated to:
// cublasCherkEx-NEXT:   dpct::blas::syherk<true>(handle, uplo, trans, n, k, alpha, a, a_type, lda, beta, c, c_type, ldc, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCopyEx | FileCheck %s -check-prefix=cublasCopyEx
// cublasCopyEx: CUDA API:
// cublasCopyEx-NEXT:   cublasCopyEx(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
// cublasCopyEx-NEXT:                x_type /*cudaDataType*/, incx /*int*/, y /*void **/,
// cublasCopyEx-NEXT:                y_type /*cudaDataType*/, incy /*int*/);
// cublasCopyEx-NEXT: Is migrated to:
// cublasCopyEx-NEXT:   dpct::blas::copy(handle, n, x, x_type, incx, y, y_type, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyrk3mEx | FileCheck %s -check-prefix=cublasCsyrk3mEx
// cublasCsyrk3mEx: CUDA API:
// cublasCsyrk3mEx-NEXT:   cublasCsyrk3mEx(handle /*cublasHandle_t*/, uplo /*cublasFillMode_t*/,
// cublasCsyrk3mEx-NEXT:                   trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCsyrk3mEx-NEXT:                   alpha /*const cuComplex **/, a /*const void **/,
// cublasCsyrk3mEx-NEXT:                   a_type /*cudaDataType*/, lda /*int*/,
// cublasCsyrk3mEx-NEXT:                   beta /*const cuComplex **/, c /*void **/,
// cublasCsyrk3mEx-NEXT:                   c_type /*cudaDataType*/, ldc /*int*/);
// cublasCsyrk3mEx-NEXT: Is migrated to:
// cublasCsyrk3mEx-NEXT:   dpct::blas::syherk<false>(handle, uplo, trans, n, k, alpha, a, a_type, lda, beta, c, c_type, ldc, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyrkEx | FileCheck %s -check-prefix=cublasCsyrkEx
// cublasCsyrkEx: CUDA API:
// cublasCsyrkEx-NEXT:   cublasCsyrkEx(handle /*cublasHandle_t*/, uplo /*cublasFillMode_t*/,
// cublasCsyrkEx-NEXT:                 trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCsyrkEx-NEXT:                 alpha /*const cuComplex **/, a /*const void **/,
// cublasCsyrkEx-NEXT:                 a_type /*cudaDataType*/, lda /*int*/,
// cublasCsyrkEx-NEXT:                 beta /*const cuComplex **/, c /*void **/,
// cublasCsyrkEx-NEXT:                 c_type /*cudaDataType*/, ldc /*int*/);
// cublasCsyrkEx-NEXT: Is migrated to:
// cublasCsyrkEx-NEXT:   dpct::blas::syherk<false>(handle, uplo, trans, n, k, alpha, a, a_type, lda, beta, c, c_type, ldc, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIamaxEx | FileCheck %s -check-prefix=cublasIamaxEx
// cublasIamaxEx: CUDA API:
// cublasIamaxEx-NEXT:   cublasIamaxEx(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
// cublasIamaxEx-NEXT:                 x_type /*cudaDataType*/, incx /*int*/, result /*int **/);
// cublasIamaxEx-NEXT: Is migrated to:
// cublasIamaxEx-NEXT:   dpct::blas::iamax(handle, n, x, x_type, incx, result);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIaminEx | FileCheck %s -check-prefix=cublasIaminEx
// cublasIaminEx: CUDA API:
// cublasIaminEx-NEXT:   cublasIaminEx(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
// cublasIaminEx-NEXT:                 x_type /*cudaDataType*/, incx /*int*/, result /*int **/);
// cublasIaminEx-NEXT: Is migrated to:
// cublasIaminEx-NEXT:   dpct::blas::iamin(handle, n, x, x_type, incx, result);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasRotmEx | FileCheck %s -check-prefix=cublasRotmEx
// cublasRotmEx: CUDA API:
// cublasRotmEx-NEXT:   cublasRotmEx(handle /*cublasHandle_t*/, n /*int*/, x /*void **/,
// cublasRotmEx-NEXT:                x_type /*cudaDataType*/, incx /*int*/, y /*void **/,
// cublasRotmEx-NEXT:                y_type /*cudaDataType*/, incy /*int*/, param /*const void **/,
// cublasRotmEx-NEXT:                param_type /*cudaDataType*/, execution_type /*cudaDataType*/);
// cublasRotmEx-NEXT: Is migrated to:
// cublasRotmEx-NEXT:   dpct::blas::rotm(handle, n, x, x_type, incx, y, y_type, incy, param, param_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSwapEx | FileCheck %s -check-prefix=cublasSwapEx
// cublasSwapEx: CUDA API:
// cublasSwapEx-NEXT:   cublasSwapEx(handle /*cublasHandle_t*/, n /*int*/, x /*void **/,
// cublasSwapEx-NEXT:                x_type /*cudaDataType*/, incx /*int*/, y /*void **/,
// cublasSwapEx-NEXT:                y_type /*cudaDataType*/, incy /*int*/);
// cublasSwapEx-NEXT: Is migrated to:
// cublasSwapEx-NEXT:   dpct::blas::swap(handle, n, x, x_type, incx, y, y_type, incy);
