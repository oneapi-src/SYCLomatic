#include "cusparse.h"

void test(cusparseHandle_t handle, int m, int n, int k, const float *alpha,
          const cusparseMatDescr_t descr_a, int nnz_a, const float *value_a,
          const int *row_ptr_a, const int *col_idx_a,
          const cusparseMatDescr_t descr_b, int nnz_b, const float *value_b,
          const int *row_ptr_b, const int *col_idx_b, const float *beta,
          const cusparseMatDescr_t descr_d, int nnz_d, const float *value_d,
          const int *row_ptr_d, const int *col_idx_d,
          const cusparseMatDescr_t descr_c, float *value_c,
          const int *row_ptr_c, int *col_idx_c, const csrgemm2Info_t info,
          void *buffer) {
  // Start
  cusparseScsrgemm2(
      handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const float **/, descr_a /*const cusparseMatDescr_t*/,
      nnz_a /*int*/, value_a /*const float **/, row_ptr_a /*const int **/,
      col_idx_a /*const int **/, descr_b /*const cusparseMatDescr_t*/,
      nnz_b /*int*/, value_b /*const float **/, row_ptr_b /*const int **/,
      col_idx_b /*const int **/, beta /*const float **/,
      descr_d /*const cusparseMatDescr_t*/, nnz_d /*int*/,
      value_d /*const float **/, row_ptr_d /*const int **/,
      col_idx_d /*const int **/, descr_c /*const cusparseMatDescr_t*/,
      value_c /*float **/, row_ptr_c /*const int **/, col_idx_c /*int **/,
      info /*const csrgemm2Info_t*/, buffer /*void **/);
  // End
}
