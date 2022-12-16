// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none   --use-custom-helper=api -out-root %T/BlasUtils/api_test25_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test25_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test25_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test25_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test25_out

// CHECK: 14

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_gemm_batch

int main() {
  cublasHandle_t handle;
  void * alpha;
  void * beta;
  const void** a;
  const void** b;
  void** c;

  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha, a, CUDA_R_16F, 4, b, CUDA_R_16F, 4, beta, c, CUDA_R_16F, 4, 2, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  return 0;
}
