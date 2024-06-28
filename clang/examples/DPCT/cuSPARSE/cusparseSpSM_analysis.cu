#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t op_a,
          cusparseOperation_t op_b, const void *alpha,
          cusparseSpMatDescr_t mat_a, cusparseDnMatDescr_t mat_b,
          cusparseDnMatDescr_t mat_c, cudaDataType compute_type,
          cusparseSpSMAlg_t alg, cusparseSpSMDescr_t desc, void *buffer) {
  // Start
  cusparseSpSM_analysis(
      handle /*cusparseHandle_t*/, op_a /*cusparseOperation_t*/,
      op_b /*cusparseOperation_t*/, alpha /*const void **/,
      mat_a /*cusparseSpMatDescr_t*/, mat_b /*cusparseSpMatDescr_t*/,
      mat_c /*cusparseSpMatDescr_t*/, compute_type /*cudaDataType*/,
      alg /*cusparseSpSMAlg_t*/, desc /*cusparseSpSMDescr_t*/,
      buffer /*void **/);
  // End
}
