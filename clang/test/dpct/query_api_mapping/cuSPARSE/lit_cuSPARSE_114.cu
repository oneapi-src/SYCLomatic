// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpSM_analysis | FileCheck %s -check-prefix=cusparseSpSM_analysis
// cusparseSpSM_analysis: CUDA API:
// cusparseSpSM_analysis-NEXT:   cusparseSpSM_analysis(
// cusparseSpSM_analysis-NEXT:       handle /*cusparseHandle_t*/, op_a /*cusparseOperation_t*/,
// cusparseSpSM_analysis-NEXT:       op_b /*cusparseOperation_t*/, alpha /*const void **/,
// cusparseSpSM_analysis-NEXT:       mat_a /*cusparseSpMatDescr_t*/, mat_b /*cusparseSpMatDescr_t*/,
// cusparseSpSM_analysis-NEXT:       mat_c /*cusparseSpMatDescr_t*/, compute_type /*cudaDataType*/,
// cusparseSpSM_analysis-NEXT:       alg /*cusparseSpSMAlg_t*/, desc /*cusparseSpSMDescr_t*/,
// cusparseSpSM_analysis-NEXT:       buffer /*void **/);
// cusparseSpSM_analysis-NEXT: Is migrated to:
// cusparseSpSM_analysis-NEXT:   dpct::sparse::spsm_optimize(handle->get_queue(), op_a, mat_a, mat_b, mat_c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpSM_bufferSize | FileCheck %s -check-prefix=cusparseSpSM_bufferSize
// cusparseSpSM_bufferSize: CUDA API:
// cusparseSpSM_bufferSize-NEXT:   size_t buffer_size;
// cusparseSpSM_bufferSize-NEXT:   cusparseSpSM_bufferSize(
// cusparseSpSM_bufferSize-NEXT:       handle /*cusparseHandle_t*/, op_a /*cusparseOperation_t*/,
// cusparseSpSM_bufferSize-NEXT:       op_b /*cusparseOperation_t*/, alpha /*const void **/,
// cusparseSpSM_bufferSize-NEXT:       mat_a /*cusparseSpMatDescr_t*/, mat_b /*cusparseSpMatDescr_t*/,
// cusparseSpSM_bufferSize-NEXT:       mat_c /*cusparseSpMatDescr_t*/, compute_type /*cudaDataType*/,
// cusparseSpSM_bufferSize-NEXT:       alg /*cusparseSpSMAlg_t*/, desc /*cusparseSpSMDescr_t*/,
// cusparseSpSM_bufferSize-NEXT:       &buffer_size /*size_t **/);
// cusparseSpSM_bufferSize-NEXT: Is migrated to:
// cusparseSpSM_bufferSize-NEXT:   size_t buffer_size;
// cusparseSpSM_bufferSize-NEXT:   buffer_size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpSM_createDescr | FileCheck %s -check-prefix=cusparseSpSM_createDescr
// cusparseSpSM_createDescr: CUDA API:
// cusparseSpSM_createDescr-NEXT:   cusparseSpSM_createDescr(&desc /*cusparseSpSMDescr_t **/);
// cusparseSpSM_createDescr-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpSM_destroyDescr | FileCheck %s -check-prefix=cusparseSpSM_destroyDescr
// cusparseSpSM_destroyDescr: CUDA API:
// cusparseSpSM_destroyDescr-NEXT:   cusparseSpSM_destroyDescr(desc /*cusparseSpSMDescr_t*/);
// cusparseSpSM_destroyDescr-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpSM_solve | FileCheck %s -check-prefix=cusparseSpSM_solve
// cusparseSpSM_solve: CUDA API:
// cusparseSpSM_solve-NEXT:   cusparseSpSM_solve(
// cusparseSpSM_solve-NEXT:       handle /*cusparseHandle_t*/, op_a /*cusparseOperation_t*/,
// cusparseSpSM_solve-NEXT:       op_b /*cusparseOperation_t*/, alpha /*const void **/,
// cusparseSpSM_solve-NEXT:       mat_a /*cusparseSpMatDescr_t*/, mat_b /*cusparseSpMatDescr_t*/,
// cusparseSpSM_solve-NEXT:       mat_c /*cusparseSpMatDescr_t*/, compute_type /*cudaDataType*/,
// cusparseSpSM_solve-NEXT:       alg /*cusparseSpSMAlg_t*/, desc /*cusparseSpSMDescr_t*/);
// cusparseSpSM_solve-NEXT: Is migrated to:
// cusparseSpSM_solve-NEXT:   dpct::sparse::spsm(handle->get_queue(), op_a, op_b, alpha, mat_a, mat_b, mat_c, compute_type);
