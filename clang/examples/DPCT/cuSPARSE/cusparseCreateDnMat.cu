#include "cusparse.h"

void test(int64_t rows, int64_t cols, int64_t ld, void *value,
          cudaDataType value_type, cusparseOrder_t order) {
  // Start
  cusparseDnMatDescr_t desc;
  cusparseCreateDnMat(&desc /*cusparseDnMatDescr_t **/, rows /*int64_t*/,
                      cols /*int64_t*/, ld /*int64_t*/, value /*void **/,
                      value_type /*cudaDataType*/, order /*cusparseOrder_t*/);
  // End
}
