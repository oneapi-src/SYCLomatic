#include "cusparse.h"

void test(int64_t size, void *value, cudaDataType value_type) {
  // Start
  cusparseConstDnVecDescr_t desc;
  cusparseCreateConstDnVec(&desc /*cusparseConstDnVecDescr_t **/,
                           size /*int64_t*/, value /*void **/,
                           value_type /*cudaDataType*/);
  // End
}
