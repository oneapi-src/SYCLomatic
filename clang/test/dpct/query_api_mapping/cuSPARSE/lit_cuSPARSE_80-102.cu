// UNSUPPORTED: cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6, cuda-12.8, cuda-12.9
// UNSUPPORTED: v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6, v12.8, v12.9

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseXcsrgemmNnz | FileCheck %s -check-prefix=cusparseXcsrgemmNnz
// cusparseXcsrgemmNnz: CUDA API:
// cusparseXcsrgemmNnz-NEXT:   cusparseXcsrgemmNnz(
// cusparseXcsrgemmNnz-NEXT:       handle /*cusparseHandle_t*/, trans_a /*cusparseOperation_t*/,
// cusparseXcsrgemmNnz-NEXT:       trans_b /*cusparseOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseXcsrgemmNnz-NEXT:       descr_a /*const cusparseMatDescr_t*/, nnz_a /*int*/,
// cusparseXcsrgemmNnz-NEXT:       row_ptr_a /*const int **/, col_idx_a /*const int **/,
// cusparseXcsrgemmNnz-NEXT:       descr_b /*const cusparseMatDescr_t*/, nnz_b /*int*/,
// cusparseXcsrgemmNnz-NEXT:       row_ptr_b /*const int **/, col_idx_b /*const int **/,
// cusparseXcsrgemmNnz-NEXT:       descr_c /*const cusparseMatDescr_t*/, row_ptr_c /*int **/, nnz /*int **/);
// cusparseXcsrgemmNnz-NEXT: Is migrated to:
// cusparseXcsrgemmNnz-NEXT:   dpct::sparse::csrgemm_nnz(handle, trans_a, trans_b, m, n, k, descr_a, nnz_a, dpct_placeholder, row_ptr_a, col_idx_a, descr_b, nnz_b, dpct_placeholder, row_ptr_b, col_idx_b, descr_c, row_ptr_c, nnz);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrgemm | FileCheck %s -check-prefix=cusparseScsrgemm
// cusparseScsrgemm: CUDA API:
// cusparseScsrgemm-NEXT:   cusparseScsrgemm(
// cusparseScsrgemm-NEXT:       handle /*cusparseHandle_t*/, trans_a /*cusparseOperation_t*/,
// cusparseScsrgemm-NEXT:       trans_b /*cusparseOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseScsrgemm-NEXT:       descr_a /*const cusparseMatDescr_t*/, nnz_a /*int*/,
// cusparseScsrgemm-NEXT:       value_a /*const float **/, row_ptr_a /*const int **/,
// cusparseScsrgemm-NEXT:       col_idx_a /*const int **/, descr_b /*const cusparseMatDescr_t*/,
// cusparseScsrgemm-NEXT:       nnz_b /*int*/, value_b /*const float **/, row_ptr_b /*const int **/,
// cusparseScsrgemm-NEXT:       col_idx_b /*const int **/, descr_c /*const cusparseMatDescr_t*/,
// cusparseScsrgemm-NEXT:       value_c /*float **/, row_ptr_c /*const int **/, col_idx_c /*int **/);
// cusparseScsrgemm-NEXT: Is migrated to:
// cusparseScsrgemm-NEXT:   dpct::sparse::csrgemm(handle, trans_a, trans_b, m, n, k, descr_a, value_a, row_ptr_a, col_idx_a, descr_b, value_b, row_ptr_b, col_idx_b, descr_c, value_c, row_ptr_c, col_idx_c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrgemm | FileCheck %s -check-prefix=cusparseDcsrgemm
// cusparseDcsrgemm: CUDA API:
// cusparseDcsrgemm-NEXT:   cusparseDcsrgemm(
// cusparseDcsrgemm-NEXT:       handle /*cusparseHandle_t*/, trans_a /*cusparseOperation_t*/,
// cusparseDcsrgemm-NEXT:       trans_b /*cusparseOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseDcsrgemm-NEXT:       descr_a /*const cusparseMatDescr_t*/, nnz_a /*int*/,
// cusparseDcsrgemm-NEXT:       value_a /*const double **/, row_ptr_a /*const int **/,
// cusparseDcsrgemm-NEXT:       col_idx_a /*const int **/, descr_b /*const cusparseMatDescr_t*/,
// cusparseDcsrgemm-NEXT:       nnz_b /*int*/, value_b /*const double **/, row_ptr_b /*const int **/,
// cusparseDcsrgemm-NEXT:       col_idx_b /*const int **/, descr_c /*const cusparseMatDescr_t*/,
// cusparseDcsrgemm-NEXT:       value_c /*double **/, row_ptr_c /*const int **/, col_idx_c /*int **/);
// cusparseDcsrgemm-NEXT: Is migrated to:
// cusparseDcsrgemm-NEXT:   dpct::sparse::csrgemm(handle, trans_a, trans_b, m, n, k, descr_a, value_a, row_ptr_a, col_idx_a, descr_b, value_b, row_ptr_b, col_idx_b, descr_c, value_c, row_ptr_c, col_idx_c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrgemm | FileCheck %s -check-prefix=cusparseCcsrgemm
// cusparseCcsrgemm: CUDA API:
// cusparseCcsrgemm-NEXT:   cusparseCcsrgemm(
// cusparseCcsrgemm-NEXT:       handle /*cusparseHandle_t*/, trans_a /*cusparseOperation_t*/,
// cusparseCcsrgemm-NEXT:       trans_b /*cusparseOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseCcsrgemm-NEXT:       descr_a /*const cusparseMatDescr_t*/, nnz_a /*int*/,
// cusparseCcsrgemm-NEXT:       value_a /*const cuComplex **/, row_ptr_a /*const int **/,
// cusparseCcsrgemm-NEXT:       col_idx_a /*const int **/, descr_b /*const cusparseMatDescr_t*/,
// cusparseCcsrgemm-NEXT:       nnz_b /*int*/, value_b /*const cuComplex **/, row_ptr_b /*const int **/,
// cusparseCcsrgemm-NEXT:       col_idx_b /*const int **/, descr_c /*const cusparseMatDescr_t*/,
// cusparseCcsrgemm-NEXT:       value_c /*cuComplex **/, row_ptr_c /*const int **/, col_idx_c /*int **/);
// cusparseCcsrgemm-NEXT: Is migrated to:
// cusparseCcsrgemm-NEXT:   dpct::sparse::csrgemm(handle, trans_a, trans_b, m, n, k, descr_a, value_a, row_ptr_a, col_idx_a, descr_b, value_b, row_ptr_b, col_idx_b, descr_c, value_c, row_ptr_c, col_idx_c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrgemm | FileCheck %s -check-prefix=cusparseZcsrgemm
// cusparseZcsrgemm: CUDA API:
// cusparseZcsrgemm-NEXT:   cusparseZcsrgemm(
// cusparseZcsrgemm-NEXT:       handle /*cusparseHandle_t*/, trans_a /*cusparseOperation_t*/,
// cusparseZcsrgemm-NEXT:       trans_b /*cusparseOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseZcsrgemm-NEXT:       descr_a /*const cusparseMatDescr_t*/, nnz_a /*int*/,
// cusparseZcsrgemm-NEXT:       value_a /*const cuDoubleComplex **/, row_ptr_a /*const int **/,
// cusparseZcsrgemm-NEXT:       col_idx_a /*const int **/, descr_b /*const cusparseMatDescr_t*/,
// cusparseZcsrgemm-NEXT:       nnz_b /*int*/, value_b /*const cuDoubleComplex **/,
// cusparseZcsrgemm-NEXT:       row_ptr_b /*const int **/, col_idx_b /*const int **/,
// cusparseZcsrgemm-NEXT:       descr_c /*const cusparseMatDescr_t*/, value_c /*cuDoubleComplex **/,
// cusparseZcsrgemm-NEXT:       row_ptr_c /*const int **/, col_idx_c /*int **/);
// cusparseZcsrgemm-NEXT: Is migrated to:
// cusparseZcsrgemm-NEXT:   dpct::sparse::csrgemm(handle, trans_a, trans_b, m, n, k, descr_a, value_a, row_ptr_a, col_idx_a, descr_b, value_b, row_ptr_b, col_idx_b, descr_c, value_c, row_ptr_c, col_idx_c);
