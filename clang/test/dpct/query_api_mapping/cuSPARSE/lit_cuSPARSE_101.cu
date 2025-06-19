// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCreateCoo | FileCheck %s -check-prefix=cusparseCreateCoo
// cusparseCreateCoo: CUDA API:
// cusparseCreateCoo-NEXT:   cusparseSpMatDescr_t desc;
// cusparseCreateCoo-NEXT:   cusparseCreateCoo(
// cusparseCreateCoo-NEXT:       &desc /*cusparseSpMatDescr_t **/, rows /*int64_t*/, cols /*int64_t*/,
// cusparseCreateCoo-NEXT:       nnz /*int64_t*/, row_ptr /*void **/, col_idx /*void **/, value /*void **/,
// cusparseCreateCoo-NEXT:       idx_type /*cusparseIndexType_t*/, idx_base /*cusparseIndexBase_t*/,
// cusparseCreateCoo-NEXT:       value_type /*cudaDataType*/);
// cusparseCreateCoo-NEXT: Is migrated to:
// cusparseCreateCoo-NEXT:   dpct::sparse::sparse_matrix_desc_t desc;
// cusparseCreateCoo-NEXT:   desc = std::make_shared<dpct::sparse::sparse_matrix_desc>(rows, cols, nnz, row_ptr, col_idx, value, idx_type, idx_type, idx_base, value_type, dpct::sparse::matrix_format::coo);
