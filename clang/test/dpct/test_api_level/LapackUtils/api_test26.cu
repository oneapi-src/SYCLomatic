// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test26_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test26_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test26_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test26_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test26_out

// CHECK: 31
// TEST_FEATURE: LapackUtils_syheevd_scratchpad_size

#include "cusolverDn.h"

int main() {
  float *a_s;
  float *w_s;
  cusolverDnHandle_t handle;
  cusolverDnParams_t params;
  size_t lwork_s;
  size_t lwork_host_s;
  cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, w_s, CUDA_R_32F, &lwork_s, &lwork_host_s);
  return 0;
}
