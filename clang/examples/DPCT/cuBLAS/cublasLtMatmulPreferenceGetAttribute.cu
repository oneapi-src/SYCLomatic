#include "cublasLt.h"

void test(cublasLtMatmulPreference_t pref,
          cublasLtMatmulPreferenceAttributes_t attr, void *buf,
          size_t size_in_bytes, size_t *size_written) {
  // Start
  cublasLtMatmulPreferenceGetAttribute(
      pref /*cublasLtMatmulPreference_t*/,
      attr /*cublasLtMatmulPreferenceAttributes_t*/, buf /*void **/,
      size_in_bytes /*size_t*/, size_written /*size_t **/);
  // End
}
