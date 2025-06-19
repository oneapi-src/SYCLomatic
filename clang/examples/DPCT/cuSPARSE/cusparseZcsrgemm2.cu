#include "cusparse.h"

void test(cusparseHandle_t handle, int m, int n, int k,
          const cuDoubleComplex *alpha, const cusparseMatDescr_t descr_a,
          int nnz_a, const cuDoubleComplex *value_a, const int *row_ptr_a,
          const int *col_idx_a, const cusparseMatDescr_t descr_b, int nnz_b,
          const cuDoubleComplex *value_b, const int *row_ptr_b,
          const int *col_idx_b, const cuDoubleComplex *beta,
          const cusparseMatDescr_t descr_d, int nnz_d,
          const cuDoubleComplex *value_d, const int *row_ptr_d,
          const int *col_idx_d, const cusparseMatDescr_t descr_c,
          cuDoubleComplex *value_c, const int *row_ptr_c, int *col_idx_c,
          const csrgemm2Info_t info, void *buffer) {
  // Start
  cusparseZcsrgemm2(
      handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const cuDoubleComplex **/, descr_a /*const cusparseMatDescr_t*/,
      nnz_a /*int*/, value_a /*const cuDoubleComplex **/,
      row_ptr_a /*const int **/, col_idx_a /*const int **/,
      descr_b /*const cusparseMatDescr_t*/, nnz_b /*int*/,
      value_b /*const cuDoubleComplex **/, row_ptr_b /*const int **/,
      col_idx_b /*const int **/, beta /*const cuDoubleComplex **/,
      descr_d /*const cusparseMatDescr_t*/, nnz_d /*int*/,
      value_d /*const cuDoubleComplex **/, row_ptr_d /*const int **/,
      col_idx_d /*const int **/, descr_c /*const cusparseMatDescr_t*/,
      value_c /*cuDoubleComplex **/, row_ptr_c /*const int **/,
      col_idx_c /*int **/, info /*const csrgemm2Info_t*/, buffer /*void **/);
  // End
}
