// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: mkdir %T/fix_internal_error_3
// RUN: dpct --out-root %T/fix_internal_error_3 %s --cuda-include-path="%cuda-path/include" --extra-arg="-I%S" > %T/fix_internal_error_3/output.txt 2>&1 || true
// RUN: grep "dpct internal error" %T/fix_internal_error_3/output.txt | wc -l > %T/fix_internal_error_3/wc_output.txt || true
// RUN: FileCheck %s --match-full-lines --input-file %T/fix_internal_error_3/wc_output.txt
// RUN: rm -rf %T/fix_internal_error_3

// CHECK: 0

#include "fix_internal_error_3.h"

template <typename T> void foo(int a, int b) {
  cublasHandle_t handle;
  cudaDataType_t AType;
  cudaDataType_t BType;
  cudaDataType_t CType;
  cudaDataType_t computeType;
  float alpha = (float)1.0f;
  float beta = (float)0.0f;
  T *d_A;
  T *d_B;
  T *d_C;
  cublasGemmAlgo_t algo;
  cublasStatus_t status;

  status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, a, a, b,
                                      &alpha, d_B, BType, b, a * b, d_A, AType,
                                      b, a * b, &beta, d_C, CUDA_R_32F, a,
                                      a * a, 123, computeType, algo);
}

template void foo<float>(int a, int b);
