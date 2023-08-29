// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGemmBatchedEx | FileCheck %s -check-prefix=cublasGemmBatchedEx
// cublasGemmBatchedEx: CUDA API:
// cublasGemmBatchedEx-NEXT:   cublasGemmBatchedEx(
// cublasGemmBatchedEx-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasGemmBatchedEx-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasGemmBatchedEx-NEXT:       alpha /*const void **/, a /*const void *const **/, atype /*cudaDataType*/,
// cublasGemmBatchedEx-NEXT:       lda /*int*/, b /*const void *const **/, btype /*cudaDataType*/,
// cublasGemmBatchedEx-NEXT:       ldb /*int*/, beta /*const void **/, c /*void *const **/,
// cublasGemmBatchedEx-NEXT:       ctype /*cudaDataType*/, ldc /*int*/, group_count /*int*/,
// cublasGemmBatchedEx-NEXT:       computetype_computeType_t /*cublasComputeType_t*/,
// cublasGemmBatchedEx-NEXT:       algo /*cublasGemmAlgo_t*/);
// cublasGemmBatchedEx-NEXT:   cublasGemmBatchedEx(
// cublasGemmBatchedEx-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasGemmBatchedEx-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasGemmBatchedEx-NEXT:       alpha /*const void **/, a /*const void *const **/, atype /*cudaDataType*/,
// cublasGemmBatchedEx-NEXT:       lda /*int*/, b /*const void *const **/, btype /*cudaDataType*/,
// cublasGemmBatchedEx-NEXT:       ldb /*int*/, beta /*const void **/, c /*void *const **/,
// cublasGemmBatchedEx-NEXT:       ctype /*cudaDataType*/, ldc /*int*/, group_count /*int*/,
// cublasGemmBatchedEx-NEXT:       computetype_dataType /*cudaDataType*/, algo /*cublasGemmAlgo_t*/);
// cublasGemmBatchedEx-NEXT: Is migrated to:
// cublasGemmBatchedEx-NEXT:   dpct::gemm_batch(handle->get_queue(), transa, transb, m, n, k, alpha, const_cast<void const **>(a), atype, lda, const_cast<void const **>(b), btype, ldb, beta, const_cast<void **>(c), ctype, ldc, group_count, computetype_computeType_t);
// cublasGemmBatchedEx-NEXT:   dpct::gemm_batch(handle->get_queue(), transa, transb, m, n, k, alpha, const_cast<void const **>(a), atype, lda, const_cast<void const **>(b), btype, ldb, beta, const_cast<void **>(c), ctype, ldc, group_count, computetype_dataType);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGemmEx | FileCheck %s -check-prefix=cublasGemmEx
// cublasGemmEx: CUDA API:
// cublasGemmEx-NEXT:   cublasGemmEx(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasGemmEx-NEXT:                transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasGemmEx-NEXT:                alpha /*const void **/, a /*const void **/,
// cublasGemmEx-NEXT:                atype /*cudaDataType*/, lda /*int*/, b /*const void **/,
// cublasGemmEx-NEXT:                btype /*cudaDataType*/, ldb /*int*/, beta /*const void **/,
// cublasGemmEx-NEXT:                c /*void **/, ctype /*cudaDataType*/, ldc /*int*/,
// cublasGemmEx-NEXT:                computetype_computeType_t /*cublasComputeType_t*/,
// cublasGemmEx-NEXT:                algo /*cublasGemmAlgo_t*/);
// cublasGemmEx-NEXT:   cublasGemmEx(
// cublasGemmEx-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasGemmEx-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasGemmEx-NEXT:       alpha /*const void **/, a /*const void **/, atype /*cudaDataType*/,
// cublasGemmEx-NEXT:       lda /*int*/, b /*const void **/, btype /*cudaDataType*/, ldb /*int*/,
// cublasGemmEx-NEXT:       beta /*const void **/, c /*void **/, ctype /*cudaDataType*/, ldc /*int*/,
// cublasGemmEx-NEXT:       computetype_dataType /*cudaDataType*/, algo /*cublasGemmAlgo_t*/);
// cublasGemmEx-NEXT: Is migrated to:
// cublasGemmEx-NEXT:   dpct::gemm(handle->get_queue(), transa, transb, m, n, k, alpha, a, atype, lda, b, btype, ldb, beta, c, ctype, ldc, computetype_computeType_t);
// cublasGemmEx-NEXT:   dpct::gemm(handle->get_queue(), transa, transb, m, n, k, alpha, a, atype, lda, b, btype, ldb, beta, c, ctype, ldc, computetype_dataType);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGemmStridedBatchedEx | FileCheck %s -check-prefix=cublasGemmStridedBatchedEx
// cublasGemmStridedBatchedEx: CUDA API:
// cublasGemmStridedBatchedEx-NEXT:   cublasGemmStridedBatchedEx(
// cublasGemmStridedBatchedEx-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasGemmStridedBatchedEx-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasGemmStridedBatchedEx-NEXT:       alpha /*const void **/, a /*const void **/, atype /*cudaDataType*/,
// cublasGemmStridedBatchedEx-NEXT:       lda /*int*/, stridea /*long long int*/, b /*const void **/,
// cublasGemmStridedBatchedEx-NEXT:       btype /*cudaDataType*/, ldb /*int*/, strideb /*long long int*/,
// cublasGemmStridedBatchedEx-NEXT:       beta /*const void **/, c /*void **/, ctype /*cudaDataType*/, ldc /*int*/,
// cublasGemmStridedBatchedEx-NEXT:       stridec /*long long int*/, group_count /*int*/,
// cublasGemmStridedBatchedEx-NEXT:       computetype_computeType_t /*cublasComputeType_t*/,
// cublasGemmStridedBatchedEx-NEXT:       algo /*cublasGemmAlgo_t*/);
// cublasGemmStridedBatchedEx-NEXT:   cublasGemmStridedBatchedEx(
// cublasGemmStridedBatchedEx-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasGemmStridedBatchedEx-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasGemmStridedBatchedEx-NEXT:       alpha /*const void **/, a /*const void **/, atype /*cudaDataType*/,
// cublasGemmStridedBatchedEx-NEXT:       lda /*int*/, stridea /*long long int*/, b /*const void **/,
// cublasGemmStridedBatchedEx-NEXT:       btype /*cudaDataType*/, ldb /*int*/, strideb /*long long int*/,
// cublasGemmStridedBatchedEx-NEXT:       beta /*const void **/, c /*void **/, ctype /*cudaDataType*/, ldc /*int*/,
// cublasGemmStridedBatchedEx-NEXT:       stridec /*long long int*/, group_count /*int*/,
// cublasGemmStridedBatchedEx-NEXT:       computetype_dataType /*cudaDataType*/, algo /*cublasGemmAlgo_t*/);
// cublasGemmStridedBatchedEx-NEXT: Is migrated to:
// cublasGemmStridedBatchedEx-NEXT:   dpct::gemm_batch(handle->get_queue(), transa, transb, m, n, k, alpha, a, atype, lda, stridea, b, btype, ldb, strideb, beta, c, ctype, ldc, stridec, group_count, computetype_computeType_t);
// cublasGemmStridedBatchedEx-NEXT:   dpct::gemm_batch(handle->get_queue(), transa, transb, m, n, k, alpha, a, atype, lda, stridea, b, btype, ldb, strideb, beta, c, ctype, ldc, stridec, group_count, computetype_dataType);
