#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans_a,
          cusparseOperation_t trans_b, int m, int n, int k,
          const cusparseMatDescr_t descr_a, int nnz_a,
          const cuDoubleComplex *value_a, const int *row_ptr_a,
          const int *col_idx_a, const cusparseMatDescr_t descr_b, int nnz_b,
          const cuDoubleComplex *value_b, const int *row_ptr_b,
          const int *col_idx_b, const cusparseMatDescr_t descr_c,
          cuDoubleComplex *value_c, const int *row_ptr_c, int *col_idx_c) {
  // Start
  cusparseZcsrgemm(
      handle /*cusparseHandle_t*/, trans_a /*cusparseOperation_t*/,
      trans_b /*cusparseOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      descr_a /*const cusparseMatDescr_t*/, nnz_a /*int*/,
      value_a /*const cuDoubleComplex **/, row_ptr_a /*const int **/,
      col_idx_a /*const int **/, descr_b /*const cusparseMatDescr_t*/,
      nnz_b /*int*/, value_b /*const cuDoubleComplex **/,
      row_ptr_b /*const int **/, col_idx_b /*const int **/,
      descr_c /*const cusparseMatDescr_t*/, value_c /*cuDoubleComplex **/,
      row_ptr_c /*const int **/, col_idx_c /*int **/);
  // End
}
