// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test10_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test10_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test10_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test10_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test10_out

// CHECK: 32

// TEST_FEATURE: Util_matrix_mem_copy

#include "cublas_v2.h"

int main() {
  float* a;
  cublasSetVector(10, sizeof(float), a, 1, a, 1);
  return 0;
}
