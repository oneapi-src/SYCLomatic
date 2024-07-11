// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpSV_analysis | FileCheck %s -check-prefix=cusparseSpSV_analysis
// cusparseSpSV_analysis: CUDA API:
// cusparseSpSV_analysis-NEXT:   cusparseSpSV_analysis(
// cusparseSpSV_analysis-NEXT:       handle /*cusparseHandle_t*/, op_a /*cusparseOperation_t*/,
// cusparseSpSV_analysis-NEXT:       alpha /*const void **/, mat_a /*cusparseSpMatDescr_t*/,
// cusparseSpSV_analysis-NEXT:       vec_x /*cusparseDnVecDescr_t*/, vec_y /*cusparseDnVecDescr_t*/,
// cusparseSpSV_analysis-NEXT:       compute_type /*cudaDataType*/, alg /*cusparseSpSVAlg_t*/,
// cusparseSpSV_analysis-NEXT:       desc /*cusparseSpSVDescr_t*/, buffer /*void **/);
// cusparseSpSV_analysis-NEXT: Is migrated to:
// cusparseSpSV_analysis-NEXT:   dpct::sparse::spsv_optimize(handle->get_queue(), op_a, mat_a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpSV_bufferSize | FileCheck %s -check-prefix=cusparseSpSV_bufferSize
// cusparseSpSV_bufferSize: CUDA API:
// cusparseSpSV_bufferSize-NEXT:   size_t buffer_size;
// cusparseSpSV_bufferSize-NEXT:   cusparseSpSV_bufferSize(
// cusparseSpSV_bufferSize-NEXT:       handle /*cusparseHandle_t*/, op_a /*cusparseOperation_t*/,
// cusparseSpSV_bufferSize-NEXT:       alpha /*const void **/, mat_a /*cusparseSpMatDescr_t*/,
// cusparseSpSV_bufferSize-NEXT:       vec_x /*cusparseDnVecDescr_t*/, vec_y /*cusparseDnVecDescr_t*/,
// cusparseSpSV_bufferSize-NEXT:       compute_type /*cudaDataType*/, alg /*cusparseSpSVAlg_t*/,
// cusparseSpSV_bufferSize-NEXT:       desc /*cusparseSpSVDescr_t*/, &buffer_size /*size_t **/);
// cusparseSpSV_bufferSize-NEXT: Is migrated to:
// cusparseSpSV_bufferSize-NEXT:   size_t buffer_size;
// cusparseSpSV_bufferSize-NEXT:   buffer_size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpSV_createDescr | FileCheck %s -check-prefix=cusparseSpSV_createDescr
// cusparseSpSV_createDescr: CUDA API:
// cusparseSpSV_createDescr-NEXT:   cusparseSpSV_createDescr(&desc /*cusparseSpSVDescr_t **/);
// cusparseSpSV_createDescr-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpSV_destroyDescr | FileCheck %s -check-prefix=cusparseSpSV_destroyDescr
// cusparseSpSV_destroyDescr: CUDA API:
// cusparseSpSV_destroyDescr-NEXT:   cusparseSpSV_destroyDescr(desc /*cusparseSpSVDescr_t*/);
// cusparseSpSV_destroyDescr-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpSV_solve | FileCheck %s -check-prefix=cusparseSpSV_solve
// cusparseSpSV_solve: CUDA API:
// cusparseSpSV_solve-NEXT:   cusparseSpSV_solve(handle /*cusparseHandle_t*/, op_a /*cusparseOperation_t*/,
// cusparseSpSV_solve-NEXT:                      alpha /*const void **/, mat_a /*cusparseSpMatDescr_t*/,
// cusparseSpSV_solve-NEXT:                      vec_x /*cusparseDnVecDescr_t*/,
// cusparseSpSV_solve-NEXT:                      vec_y /*cusparseDnVecDescr_t*/,
// cusparseSpSV_solve-NEXT:                      compute_type /*cudaDataType*/, alg /*cusparseSpSVAlg_t*/,
// cusparseSpSV_solve-NEXT:                      desc /*cusparseSpSVDescr_t*/);
// cusparseSpSV_solve-NEXT: Is migrated to:
// cusparseSpSV_solve-NEXT:   dpct::sparse::spsv(handle->get_queue(), op_a, alpha, mat_a, vec_x, vec_y, compute_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMatGetAttribute | FileCheck %s -check-prefix=cusparseSpMatGetAttribute
// cusparseSpMatGetAttribute: CUDA API:
// cusparseSpMatGetAttribute-NEXT:   cusparseSpMatGetAttribute(desc /*cusparseSpMatDescr_t*/,
// cusparseSpMatGetAttribute-NEXT:                             attr /*cusparseSpMatAttribute_t*/, data /*void **/,
// cusparseSpMatGetAttribute-NEXT:                             data_size /*size_t*/);
// cusparseSpMatGetAttribute-NEXT: Is migrated to:
// cusparseSpMatGetAttribute-NEXT:   desc->get_attribute(attr, data, data_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMatSetAttribute | FileCheck %s -check-prefix=cusparseSpMatSetAttribute
// cusparseSpMatSetAttribute: CUDA API:
// cusparseSpMatSetAttribute-NEXT:   cusparseSpMatSetAttribute(desc /*cusparseSpMatDescr_t*/,
// cusparseSpMatSetAttribute-NEXT:                             attr /*cusparseSpMatAttribute_t*/, data /*void **/,
// cusparseSpMatSetAttribute-NEXT:                             data_size /*size_t*/);
// cusparseSpMatSetAttribute-NEXT: Is migrated to:
// cusparseSpMatSetAttribute-NEXT:   desc->set_attribute(attr, data, data_size);
