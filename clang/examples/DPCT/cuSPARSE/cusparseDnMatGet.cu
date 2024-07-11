#include "cusparse.h"

void test(cusparseDnMatDescr_t desc) {
  // Start
  int64_t rows;
  int64_t cols;
  int64_t ld;
  void *value;
  cudaDataType value_type;
  cusparseOrder_t order;
  cusparseDnMatGet(desc /*cusparseDnMatDescr_t*/, &rows /*int64_t **/,
                   &cols /*int64_t **/, &ld /*int64_t **/, &value /*void ***/,
                   &value_type /*cudaDataType **/,
                   &order /*cusparseOrder_t **/);
  // End
}
