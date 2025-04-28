#include "cublasLt.h"

void test(cublasLtMatmulPreference_t pref) {
  // Start
  cublasLtMatmulPreferenceDestroy(pref /*cublasLtMatmulPreference_t*/);
  // End
}
