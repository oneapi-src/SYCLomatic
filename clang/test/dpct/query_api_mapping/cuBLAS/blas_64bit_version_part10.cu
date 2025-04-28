// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsbmv_64 | FileCheck %s -check-prefix=cublasSsbmv_64
// cublasSsbmv_64: CUDA API:
// cublasSsbmv_64-NEXT:   cublasSsbmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsbmv_64-NEXT:                  n /*int64_t*/, k /*int64_t*/, alpha /*const float **/,
// cublasSsbmv_64-NEXT:                  a /*const float **/, lda /*int64_t*/, x /*const float **/,
// cublasSsbmv_64-NEXT:                  incx /*int64_t*/, beta /*const float **/, y /*float **/,
// cublasSsbmv_64-NEXT:                  incy /*int64_t*/);
// cublasSsbmv_64-NEXT: Is migrated to:
// cublasSsbmv_64-NEXT:   oneapi::mkl::blas::column_major::sbmv(handle->get_queue(), upper_lower, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDspmv_64 | FileCheck %s -check-prefix=cublasDspmv_64
// cublasDspmv_64: CUDA API:
// cublasDspmv_64-NEXT:   cublasDspmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDspmv_64-NEXT:                  n /*int64_t*/, alpha /*const double **/, a /*const double **/,
// cublasDspmv_64-NEXT:                  x /*const double **/, incx /*int64_t*/,
// cublasDspmv_64-NEXT:                  beta /*const double **/, y /*double **/, incy /*int64_t*/);
// cublasDspmv_64-NEXT: Is migrated to:
// cublasDspmv_64-NEXT:   oneapi::mkl::blas::column_major::spmv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), a, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSspmv_64 | FileCheck %s -check-prefix=cublasSspmv_64
// cublasSspmv_64: CUDA API:
// cublasSspmv_64-NEXT:   cublasSspmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSspmv_64-NEXT:                  n /*int64_t*/, alpha /*const float **/, a /*const float **/,
// cublasSspmv_64-NEXT:                  x /*const float **/, incx /*int64_t*/, beta /*const float **/,
// cublasSspmv_64-NEXT:                  y /*float **/, incy /*int64_t*/);
// cublasSspmv_64-NEXT: Is migrated to:
// cublasSspmv_64-NEXT:   oneapi::mkl::blas::column_major::spmv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), a, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDspr2_64 | FileCheck %s -check-prefix=cublasDspr2_64
// cublasDspr2_64: CUDA API:
// cublasDspr2_64-NEXT:   cublasDspr2_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDspr2_64-NEXT:                  n /*int64_t*/, alpha /*const double **/, x /*const double **/,
// cublasDspr2_64-NEXT:                  incx /*int64_t*/, y /*const double **/, incy /*int64_t*/,
// cublasDspr2_64-NEXT:                  a /*double **/);
// cublasDspr2_64-NEXT: Is migrated to:
// cublasDspr2_64-NEXT:   oneapi::mkl::blas::column_major::spr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, y, incy, a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSspr2_64 | FileCheck %s -check-prefix=cublasSspr2_64
// cublasSspr2_64: CUDA API:
// cublasSspr2_64-NEXT:   cublasSspr2_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSspr2_64-NEXT:                  n /*int64_t*/, alpha /*const float **/, x /*const float **/,
// cublasSspr2_64-NEXT:                  incx /*int64_t*/, y /*const float **/, incy /*int64_t*/,
// cublasSspr2_64-NEXT:                  a /*float **/);
// cublasSspr2_64-NEXT: Is migrated to:
// cublasSspr2_64-NEXT:   oneapi::mkl::blas::column_major::spr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, y, incy, a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDspr_64 | FileCheck %s -check-prefix=cublasDspr_64
// cublasDspr_64: CUDA API:
// cublasDspr_64-NEXT:   cublasDspr_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDspr_64-NEXT:                 n /*int64_t*/, alpha /*const double **/, x /*const double **/,
// cublasDspr_64-NEXT:                 incx /*int64_t*/, a /*double **/);
// cublasDspr_64-NEXT: Is migrated to:
// cublasDspr_64-NEXT:   oneapi::mkl::blas::column_major::spr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSspr_64 | FileCheck %s -check-prefix=cublasSspr_64
// cublasSspr_64: CUDA API:
// cublasSspr_64-NEXT:   cublasSspr_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSspr_64-NEXT:                 n /*int64_t*/, alpha /*const float **/, x /*const float **/,
// cublasSspr_64-NEXT:                 incx /*int64_t*/, a /*float **/);
// cublasSspr_64-NEXT: Is migrated to:
// cublasSspr_64-NEXT:   oneapi::mkl::blas::column_major::spr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGemmBatchedEx_64 | FileCheck %s -check-prefix=cublasGemmBatchedEx_64
// cublasGemmBatchedEx_64: CUDA API:
// cublasGemmBatchedEx_64-NEXT:   cublasGemmBatchedEx_64(
// cublasGemmBatchedEx_64-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasGemmBatchedEx_64-NEXT:       transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasGemmBatchedEx_64-NEXT:       alpha /*const void **/, a /*const void *const **/, atype /*cudaDataType*/,
// cublasGemmBatchedEx_64-NEXT:       lda /*int64_t*/, b /*const void *const **/, btype /*cudaDataType*/,
// cublasGemmBatchedEx_64-NEXT:       ldb /*int64_t*/, beta /*const void **/, c /*void *const **/,
// cublasGemmBatchedEx_64-NEXT:       ctype /*cudaDataType*/, ldc /*int64_t*/, group_count /*int64_t*/,
// cublasGemmBatchedEx_64-NEXT:       computetype_computeType_t /*cublasComputeType_t*/,
// cublasGemmBatchedEx_64-NEXT:       algo /*cublasGemmAlgo_t*/);
// cublasGemmBatchedEx_64-NEXT: Is migrated to:
// cublasGemmBatchedEx_64-NEXT:   dpct::blas::gemm_batch(handle, transa, transb, m, n, k, alpha, const_cast<void const **>(a), atype, lda, const_cast<void const **>(b), btype, ldb, beta, const_cast<void **>(c), ctype, ldc, group_count, computetype_computeType_t);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGemmEx_64 | FileCheck %s -check-prefix=cublasGemmEx_64
// cublasGemmEx_64: CUDA API:
// cublasGemmEx_64-NEXT:   cublasGemmEx_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasGemmEx_64-NEXT:                   transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasGemmEx_64-NEXT:                   k /*int64_t*/, alpha /*const void **/, a /*const void **/,
// cublasGemmEx_64-NEXT:                   atype /*cudaDataType*/, lda /*int64_t*/, b /*const void **/,
// cublasGemmEx_64-NEXT:                   btype /*cudaDataType*/, ldb /*int64_t*/,
// cublasGemmEx_64-NEXT:                   beta /*const void **/, c /*void **/, ctype /*cudaDataType*/,
// cublasGemmEx_64-NEXT:                   ldc /*int64_t*/,
// cublasGemmEx_64-NEXT:                   computetype_computeType_t /*cublasComputeType_t*/,
// cublasGemmEx_64-NEXT:                   algo /*cublasGemmAlgo_t*/);
// cublasGemmEx_64-NEXT: Is migrated to:
// cublasGemmEx_64-NEXT:   dpct::blas::gemm(handle, transa, transb, m, n, k, alpha, a, atype, lda, b, btype, ldb, beta, c, ctype, ldc, computetype_computeType_t);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGemmStridedBatchedEx_64 | FileCheck %s -check-prefix=cublasGemmStridedBatchedEx_64
// cublasGemmStridedBatchedEx_64: CUDA API:
// cublasGemmStridedBatchedEx_64-NEXT:   cublasGemmStridedBatchedEx_64(
// cublasGemmStridedBatchedEx_64-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasGemmStridedBatchedEx_64-NEXT:       transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasGemmStridedBatchedEx_64-NEXT:       alpha /*const void **/, a /*const void **/, atype /*cudaDataType*/,
// cublasGemmStridedBatchedEx_64-NEXT:       lda /*int64_t*/, stridea /*long long int*/, b /*const void **/,
// cublasGemmStridedBatchedEx_64-NEXT:       btype /*cudaDataType*/, ldb /*int64_t*/, strideb /*long long int*/,
// cublasGemmStridedBatchedEx_64-NEXT:       beta /*const void **/, c /*void **/, ctype /*cudaDataType*/,
// cublasGemmStridedBatchedEx_64-NEXT:       ldc /*int64_t*/, stridec /*long long int*/, group_count /*int64_t*/,
// cublasGemmStridedBatchedEx_64-NEXT:       computetype_computeType_t /*cublasComputeType_t*/,
// cublasGemmStridedBatchedEx_64-NEXT:       algo /*cublasGemmAlgo_t*/);
// cublasGemmStridedBatchedEx_64-NEXT: Is migrated to:
// cublasGemmStridedBatchedEx_64-NEXT:   dpct::blas::gemm_batch(handle, transa, transb, m, n, k, alpha, a, atype, lda, stridea, b, btype, ldb, strideb, beta, c, ctype, ldc, stridec, group_count, computetype_computeType_t);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasHgemm_64 | FileCheck %s -check-prefix=cublasHgemm_64
// cublasHgemm_64: CUDA API:
// cublasHgemm_64-NEXT:   cublasHgemm_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasHgemm_64-NEXT:                  transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasHgemm_64-NEXT:                  k /*int64_t*/, alpha /*const __half **/, a /*const __half **/,
// cublasHgemm_64-NEXT:                  lda /*int64_t*/, b /*const __half **/, ldb /*int64_t*/,
// cublasHgemm_64-NEXT:                  beta /*const __half **/, c /*__half **/, ldc /*int64_t*/);
// cublasHgemm_64-NEXT: Is migrated to:
// cublasHgemm_64-NEXT:   oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIamaxEx_64 | FileCheck %s -check-prefix=cublasIamaxEx_64
// cublasIamaxEx_64: CUDA API:
// cublasIamaxEx_64-NEXT:   cublasIamaxEx_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const void **/,
// cublasIamaxEx_64-NEXT:                    x_type /*cudaDataType*/, incx /*int64_t*/,
// cublasIamaxEx_64-NEXT:                    result /*int64_t **/);
// cublasIamaxEx_64-NEXT: Is migrated to:
// cublasIamaxEx_64-NEXT:   dpct::blas::iamax(handle, n, x, x_type, incx, result);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIaminEx_64 | FileCheck %s -check-prefix=cublasIaminEx_64
// cublasIaminEx_64: CUDA API:
// cublasIaminEx_64-NEXT:   cublasIaminEx_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const void **/,
// cublasIaminEx_64-NEXT:                    x_type /*cudaDataType*/, incx /*int64_t*/,
// cublasIaminEx_64-NEXT:                    result /*int64_t **/);
// cublasIaminEx_64-NEXT: Is migrated to:
// cublasIaminEx_64-NEXT:   dpct::blas::iamin(handle, n, x, x_type, incx, result);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasNrm2Ex_64 | FileCheck %s -check-prefix=cublasNrm2Ex_64
// cublasNrm2Ex_64: CUDA API:
// cublasNrm2Ex_64-NEXT:   cublasNrm2Ex_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const void **/,
// cublasNrm2Ex_64-NEXT:                   xtype /*cudaDataType*/, incx /*int64_t*/, res /*void **/,
// cublasNrm2Ex_64-NEXT:                   restype /*cudaDataType*/, computetype /*cudaDataType*/);
// cublasNrm2Ex_64-NEXT: Is migrated to:
// cublasNrm2Ex_64-NEXT:   dpct::blas::nrm2(handle, n, x, xtype, incx, res, restype);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasRotEx_64 | FileCheck %s -check-prefix=cublasRotEx_64
// cublasRotEx_64: CUDA API:
// cublasRotEx_64-NEXT:   cublasRotEx_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*void **/,
// cublasRotEx_64-NEXT:                  xtype /*cudaDataType*/, incx /*int64_t*/, y /*void **/,
// cublasRotEx_64-NEXT:                  ytype /*cudaDataType*/, incy /*int64_t*/, c /*const void **/,
// cublasRotEx_64-NEXT:                  s /*const void **/, cstype /*cudaDataType*/,
// cublasRotEx_64-NEXT:                  computetype /*cudaDataType*/);
// cublasRotEx_64-NEXT: Is migrated to:
// cublasRotEx_64-NEXT:   dpct::blas::rot(handle, n, x, xtype, incx, y, ytype, incy, c, s, cstype);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasRotmEx_64 | FileCheck %s -check-prefix=cublasRotmEx_64
// cublasRotmEx_64: CUDA API:
// cublasRotmEx_64-NEXT:   cublasRotmEx_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*void **/,
// cublasRotmEx_64-NEXT:                   x_type /*cudaDataType*/, incx /*int64_t*/, y /*void **/,
// cublasRotmEx_64-NEXT:                   y_type /*cudaDataType*/, incy /*int64_t*/,
// cublasRotmEx_64-NEXT:                   param /*const void **/, param_type /*cudaDataType*/,
// cublasRotmEx_64-NEXT:                   execution_type /*cudaDataType*/);
// cublasRotmEx_64-NEXT: Is migrated to:
// cublasRotmEx_64-NEXT:   dpct::blas::rotm(handle, n, x, x_type, incx, y, y_type, incy, param, param_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasScalEx_64 | FileCheck %s -check-prefix=cublasScalEx_64
// cublasScalEx_64: CUDA API:
// cublasScalEx_64-NEXT:   cublasScalEx_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasScalEx_64-NEXT:                   alpha /*const void **/, alphatype /*cudaDataType*/,
// cublasScalEx_64-NEXT:                   x /*void **/, xtype /*cudaDataType*/, incx /*int64_t*/,
// cublasScalEx_64-NEXT:                   computetype /*cudaDataType*/);
// cublasScalEx_64-NEXT: Is migrated to:
// cublasScalEx_64-NEXT:   dpct::blas::scal(handle, n, alpha, alphatype, x, xtype, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgemmEx_64 | FileCheck %s -check-prefix=cublasSgemmEx_64
// cublasSgemmEx_64: CUDA API:
// cublasSgemmEx_64-NEXT:   cublasSgemmEx_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasSgemmEx_64-NEXT:                    transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasSgemmEx_64-NEXT:                    k /*int64_t*/, alpha /*const float **/, a /*const void **/,
// cublasSgemmEx_64-NEXT:                    atype /*cudaDataType*/, lda /*int64_t*/, b /*const void **/,
// cublasSgemmEx_64-NEXT:                    btype /*cudaDataType*/, ldb /*int64_t*/,
// cublasSgemmEx_64-NEXT:                    beta /*const float **/, c /*void **/, ctype /*cudaDataType*/,
// cublasSgemmEx_64-NEXT:                    ldc /*int64_t*/);
// cublasSgemmEx_64-NEXT: Is migrated to:
// cublasSgemmEx_64-NEXT:   dpct::blas::gemm(handle, transa, transb, m, n, k, alpha, a, atype, lda, b, btype, ldb, beta, c, ctype, ldc, dpct::library_data_t::real_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSwapEx_64 | FileCheck %s -check-prefix=cublasSwapEx_64
// cublasSwapEx_64: CUDA API:
// cublasSwapEx_64-NEXT:   cublasSwapEx_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*void **/,
// cublasSwapEx_64-NEXT:                   x_type /*cudaDataType*/, incx /*int64_t*/, y /*void **/,
// cublasSwapEx_64-NEXT:                   y_type /*cudaDataType*/, incy /*int64_t*/);
// cublasSwapEx_64-NEXT: Is migrated to:
// cublasSwapEx_64-NEXT:   dpct::blas::swap(handle, n, x, x_type, incx, y, y_type, incy);
