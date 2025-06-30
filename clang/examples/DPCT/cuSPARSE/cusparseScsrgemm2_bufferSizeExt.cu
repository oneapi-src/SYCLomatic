#include "cusparse.h"

void test(cusparseHandle_t handle, int m, int n, int k, const float *alpha,
          const cusparseMatDescr_t descr_a, int nnz_a, const int *row_ptr_a,
          const int *col_idx_a, const cusparseMatDescr_t descr_b, int nnz_b,
          const int *row_ptr_b, const int *col_idx_b, const float *beta,
          const cusparseMatDescr_t descr_d, int nnz_d, const int *row_ptr_d,
          const int *col_idx_d, csrgemm2Info_t info, size_t *buffer_size) {
  // Start
  cusparseScsrgemm2_bufferSizeExt(
      handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const float **/, descr_a /*const cusparseMatDescr_t*/,
      nnz_a /*int*/, row_ptr_a /*const int **/, col_idx_a /*const int **/,
      descr_b /*const cusparseMatDescr_t*/, nnz_b /*int*/,
      row_ptr_b /*const int **/, col_idx_b /*const int **/,
      beta /*const float **/, descr_d /*const cusparseMatDescr_t*/,
      nnz_d /*int*/, row_ptr_d /*const int **/, col_idx_d /*const int **/,
      info /*csrgemm2Info_t*/, buffer_size /*size_t **/);
  // End
}
