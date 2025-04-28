#include "cublasLt.h"

void test(cublasLtMatmulPreference_t pref,
          cublasLtMatmulPreferenceAttributes_t attr, const void *buf,
          size_t size_in_bytes) {
  // Start
  cublasLtMatmulPreferenceSetAttribute(
      pref /*cublasLtMatmulPreference_t*/,
      attr /*cublasLtMatmulPreferenceAttributes_t*/, buf /*const void **/,
      size_in_bytes /*size_t*/);
  // End
}
