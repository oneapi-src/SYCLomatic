#include "cublasLt.h"

void test(cublasLtMatrixLayout_t *layout, cudaDataType type, uint64_t rows,
          uint64_t cols, int64_t ld) {
  // Start
  cublasLtMatrixLayoutCreate(layout /*cublasLtMatrixLayout_t **/,
                             type /*cudaDataType*/, rows /*uint64_t*/,
                             cols /*uint64_t*/, ld /*int64_t*/);
  // End
}
