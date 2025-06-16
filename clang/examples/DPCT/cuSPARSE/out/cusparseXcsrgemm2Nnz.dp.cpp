#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/sparse_utils.hpp>

void test(dpct::sparse::descriptor_ptr handle, int m, int n, int k,
          const std::shared_ptr<dpct::sparse::matrix_info> descr_a, int nnz_a,
          const int *row_ptr_a, const int *col_idx_a,
          const std::shared_ptr<dpct::sparse::matrix_info> descr_b, int nnz_b,
          const int *row_ptr_b, const int *col_idx_b,
          const std::shared_ptr<dpct::sparse::matrix_info> descr_d, int nnz_d,
          const int *row_ptr_d, const int *col_idx_d,
          const std::shared_ptr<dpct::sparse::matrix_info> descr_c,
          int *row_ptr_c, int *nnz,
          const std::shared_ptr<dpct::sparse::csrgemm2_info> info,
          void *buffer) {
  // Start
  cusparseScsrgemm(handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
                   descr_a /*const cusparseMatDescr_t*/, nnz_a /*int*/,
                   row_ptr_a /*const int **/, col_idx_a /*const int **/,
                   descr_b /*const cusparseMatDescr_t*/, nnz_b /*int*/,
                   row_ptr_b /*const int **/, col_idx_b /*const int **/,
                   descr_d /*const cusparseMatDescr_t*/, nnz_d /*int*/,
                   row_ptr_d /*const int **/, col_idx_d /*const int **/,
                   descr_c /*const cusparseMatDescr_t*/, row_ptr_c /*int **/,
                   nnz /*int **/, info /*const csrgemm2Info_t*/,
                   buffer /*void **/);
  // End
}
