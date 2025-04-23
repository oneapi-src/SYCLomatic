#include "cublasLt.h"

void test(cublasLtMatmulDesc_t mm_esc, cublasLtMatmulDescAttributes_t attr,
          void *buf, size_t size_in_bytes) {
  // Start
  cublasLtMatmulDescSetAttribute(mm_esc /*cublasLtMatmulDesc_t*/,
                                 attr /*cublasLtMatmulDescAttributes_t*/,
                                 buf /*void **/, size_in_bytes /*size_t*/);
  // End
}
