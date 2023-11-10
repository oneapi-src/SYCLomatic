#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t transa,
          cusparseOperation_t transb, const void *alpha, cusparseSpMatDescr_t a,
          cusparseDnMatDescr_t b, const void *beta, cusparseDnMatDescr_t c,
          cudaDataType computetype, cusparseSpMMAlg_t algo, void *workspace) {
  // Start
  cusparseSpMM_preprocess(
      handle /*cusparseHandle_t*/, transa /*cusparseOperation_t*/,
      transb /*cusparseOperation_t*/, alpha /*const void **/,
      a /*cusparseSpMatDescr_t*/, b /*cusparseDnMatDescr_t*/,
      beta /*const void **/, c /*cusparseDnMatDescr_t*/,
      computetype /*cudaDataType*/, algo /*cusparseSpMMAlg_t*/,
      workspace /*void **/);
  // End
}
