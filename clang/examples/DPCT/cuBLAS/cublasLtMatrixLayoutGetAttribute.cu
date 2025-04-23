#include "cublasLt.h"

void test(cublasLtMatrixLayout_t layout, cublasLtMatrixLayoutAttribute_t attr,
          void *buf, size_t size_in_bytes, size_t *size_written) {
  // Start
  cublasLtMatrixLayoutGetAttribute(layout /*cublasLtMatrixLayout_t*/,
                                   attr /*cublasLtMatrixLayoutAttribute_t*/,
                                   buf /*void **/, size_in_bytes /*size_t*/,
                                   size_written /*size_t **/);
  // End
}
