#include "cublasLt.h"

void test(cublasLtMatmulDesc_t mm_esc, cublasLtMatmulDescAttributes_t attr,
          const void *buf, size_t size_in_bytes) {
  // Start
  cublasLtMatmulDescSetAttribute(
      mm_esc /*cublasLtMatmulDesc_t*/, attr /*cublasLtMatmulDescAttributes_t*/,
      buf /*const void **/, size_in_bytes /*size_t*/);
  // End
}
