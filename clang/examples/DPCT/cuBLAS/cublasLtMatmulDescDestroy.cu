#include "cublasLt.h"

void test(cublasLtMatmulDesc_t mm_esc) {
  // Start
  cublasLtMatmulDescDestroy(mm_esc /*cublasLtMatmulDesc_t*/);
  // End
}
