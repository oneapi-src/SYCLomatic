#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t op_a, const void *alpha,
          cusparseSpMatDescr_t mat_a, cusparseDnVecDescr_t vec_x,
          cusparseDnVecDescr_t vec_y, cudaDataType compute_type,
          cusparseSpSVAlg_t alg, cusparseSpSVDescr_t desc) {
  // Start
  size_t buffer_size;
  cusparseSpSV_bufferSize(
      handle /*cusparseHandle_t*/, op_a /*cusparseOperation_t*/,
      alpha /*const void **/, mat_a /*cusparseSpMatDescr_t*/,
      vec_x /*cusparseDnVecDescr_t*/, vec_y /*cusparseDnVecDescr_t*/,
      compute_type /*cudaDataType*/, alg /*cusparseSpSVAlg_t*/,
      desc /*cusparseSpSVDescr_t*/, &buffer_size /*size_t **/);
  // End
}
