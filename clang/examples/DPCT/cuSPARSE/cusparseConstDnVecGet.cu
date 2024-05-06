#include "cusparse.h"

void test(cusparseConstDnVecDescr_t desc) {
  // Start
  int64_t size;
  void *value;
  cudaDataType value_type;
  cusparseConstDnVecGet(desc /*cusparseConstDnVecDescr_t*/, &size /*int64_t **/,
                        &value /*void ***/, &value_type /*cudaDataType **/);
  // End
}
