// RUN: dpct --format-range=none  --use-custom-helper=api -out-root %T/Util/api_test11_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test11_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test11_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test11_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test11_out

// CHECK: 32

// TEST_FEATURE: Util_matrix_mem_copy_T

#include "cublas_v2.h"

int main() {
  cublasHandle_t handle;
  float* a;
  float *alpha;
  cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 4, 4, alpha, a, 4, a, 4, a, 4);
  return 0;
}
