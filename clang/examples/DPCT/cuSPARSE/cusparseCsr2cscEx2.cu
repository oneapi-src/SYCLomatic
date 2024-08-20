#include "cusparse.h"

void test(cusparseHandle_t handle, int m, int n, int nnz, const void *csr_value,
          const int *row_ptr, const int *col_idx, void *csc_value, int *col_ptr,
          int *row_ind, cudaDataType value_type, cusparseAction_t act,
          cusparseIndexBase_t base, cusparseCsr2CscAlg_t alg, void *buffer) {
  // Start
  cusparseCsr2cscEx2(handle /*cusparseHandle_t*/, m /*int*/, n /*int*/,
                     nnz /*int*/, csr_value /*const void **/,
                     row_ptr /*const int **/, col_idx /*const int **/,
                     csc_value /*void **/, col_ptr /*int **/, row_ind /*int **/,
                     value_type /*cudaDataType*/, act /*cusparseAction_t*/,
                     base /*cusparseIndexBase_t*/, alg /*cusparseCsr2CscAlg_t*/,
                     buffer /*void **/);
  // End
}
