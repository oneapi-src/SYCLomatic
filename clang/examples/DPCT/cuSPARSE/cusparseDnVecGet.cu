#include "cusparse.h"

void test(cusparseDnVecDescr_t desc) {
  // Start
  int64_t size;
  void *value;
  cudaDataType value_type;
  cusparseDnVecGet(desc /*cusparseDnVecDescr_t*/, &size /*int64_t **/,
                   &value /*void ***/, &value_type /*cudaDataType **/);
  // End
}
