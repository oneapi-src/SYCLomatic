#include "cusparse.h"

void test(cusparseHandle_t handle, int m, int n, int k,
          const cusparseMatDescr_t descr_a, int nnz_a, const int *row_ptr_a,
          const int *col_idx_a, const cusparseMatDescr_t descr_b, int nnz_b,
          const int *row_ptr_b, const int *col_idx_b,
          const cusparseMatDescr_t descr_d, int nnz_d, const int *row_ptr_d,
          const int *col_idx_d, const cusparseMatDescr_t descr_c,
          int *row_ptr_c, int *nnz, const csrgemm2Info_t info, void *buffer) {
  // Start
  cusparseXcsrgemm2Nnz(
      handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
      descr_a /*const cusparseMatDescr_t*/, nnz_a /*int*/,
      row_ptr_a /*const int **/, col_idx_a /*const int **/,
      descr_b /*const cusparseMatDescr_t*/, nnz_b /*int*/,
      row_ptr_b /*const int **/, col_idx_b /*const int **/,
      descr_d /*const cusparseMatDescr_t*/, nnz_d /*int*/,
      row_ptr_d /*const int **/, col_idx_d /*const int **/,
      descr_c /*const cusparseMatDescr_t*/, row_ptr_c /*int **/, nnz /*int **/,
      info /*const csrgemm2Info_t*/, buffer /*void **/);
  // End
}
