#include "cusparse.h"

void test(int64_t size, void *value, cudaDataType value_type) {
  // Start
  cusparseDnVecDescr_t desc;
  cusparseCreateDnVec(&desc /*cusparseDnVecDescr_t **/, size /*int64_t*/,
                      value /*void **/, value_type /*cudaDataType*/);
  // End
}
