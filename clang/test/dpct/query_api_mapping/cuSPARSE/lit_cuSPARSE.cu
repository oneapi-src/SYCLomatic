// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMM_preprocess | FileCheck %s -check-prefix=cusparseSpMM_preprocess
// cusparseSpMM_preprocess: CUDA API:
// cusparseSpMM_preprocess-NEXT:   cusparseSpMM_preprocess(
// cusparseSpMM_preprocess-NEXT:       handle /*cusparseHandle_t*/, transa /*cusparseOperation_t*/,
// cusparseSpMM_preprocess-NEXT:       transb /*cusparseOperation_t*/, alpha /*const void **/,
// cusparseSpMM_preprocess-NEXT:       a /*cusparseConstSpMatDescr_t*/, b /*cusparseConstDnMatDescr_t*/,
// cusparseSpMM_preprocess-NEXT:       beta /*const void **/, c /*cusparseDnMatDescr_t*/,
// cusparseSpMM_preprocess-NEXT:       computetype /*cudaDataType*/, algo /*cusparseSpMMAlg_t*/,
// cusparseSpMM_preprocess-NEXT:       workspace /*void **/);
// cusparseSpMM_preprocess-NEXT: Is migrated to:
// cusparseSpMM_preprocess-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMM_bufferSize | FileCheck %s -check-prefix=cusparseSpMM_bufferSize
// cusparseSpMM_bufferSize: CUDA API:
// cusparseSpMM_bufferSize-NEXT:   cusparseSpMM_bufferSize(
// cusparseSpMM_bufferSize-NEXT:       handle /*cusparseHandle_t*/, transa /*cusparseOperation_t*/,
// cusparseSpMM_bufferSize-NEXT:       transb /*cusparseOperation_t*/, alpha /*const void **/,
// cusparseSpMM_bufferSize-NEXT:       a /*cusparseConstSpMatDescr_t*/, b /*cusparseConstDnMatDescr_t*/,
// cusparseSpMM_bufferSize-NEXT:       beta /*const void **/, c /*cusparseDnMatDescr_t*/,
// cusparseSpMM_bufferSize-NEXT:       computetype /*cudaDataType*/, algo /*cusparseSpMMAlg_t*/,
// cusparseSpMM_bufferSize-NEXT:       workspace_size /*size_t **/);
// cusparseSpMM_bufferSize-NEXT: Is migrated to:
// cusparseSpMM_bufferSize-NEXT:   *workspace_size = 0;
// cusparseSpMM_bufferSize-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMM | FileCheck %s -check-prefix=cusparseSpMM
// cusparseSpMM: CUDA API:
// cusparseSpMM-NEXT:   cusparseSpMM(handle /*cusparseHandle_t*/, transa /*cusparseOperation_t*/,
// cusparseSpMM-NEXT:                transb /*cusparseOperation_t*/, alpha /*const void **/,
// cusparseSpMM-NEXT:                a /*cusparseConstSpMatDescr_t*/, b /*cusparseConstDnMatDescr_t*/,
// cusparseSpMM-NEXT:                beta /*const void **/, c /*cusparseDnMatDescr_t*/,
// cusparseSpMM-NEXT:                computetype /*cudaDataType*/, algo /*cusparseSpMMAlg_t*/,
// cusparseSpMM-NEXT:                workspace /*void **/);
// cusparseSpMM-NEXT: Is migrated to:
// cusparseSpMM-NEXT:   dpct::sparse::spmm(*handle, transa, transb, alpha, a, b, beta, c, computetype);
// cusparseSpMM-EMPTY:

