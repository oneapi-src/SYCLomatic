#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, const void *alpha,
          cusparseSpMatDescr_t mat_a, cusparseDnVecDescr_t vec_x,
          const void *beta, cusparseDnVecDescr_t vec_y,
          cudaDataType compute_type, cusparseSpMVAlg_t alg) {
  // Start
  size_t buffer_size;
  cusparseSpMV_bufferSize(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
      alpha /*const void **/, mat_a /*cusparseSpMatDescr_t*/,
      vec_x /*cusparseDnVecDescr_t*/, beta /*const void **/,
      vec_y /*cusparseDnVecDescr_t*/, compute_type /*cudaDataType*/,
      alg /*cusparseSpMVAlg_t*/, &buffer_size /*size_t **/);
  // End
}
