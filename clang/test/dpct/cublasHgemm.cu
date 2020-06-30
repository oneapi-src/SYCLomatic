// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1
// UNSUPPORTED: v8.0, v9.0, v9.1
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasHgemm.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

cublasHandle_t handle;
int N = 275;

int main() {

  __half *d_A_H = 0;
  __half *d_B_H = 0;
  __half *d_C_H = 0;
  __half alpha_H;
  __half beta_H;


  cublasOperation_t trans3 = CUBLAS_OP_N;
  //CHECK:mkl::blas::gemm(*handle, trans3, trans3, N, N, N, dpct::get_value(&alpha_H, *handle), d_A_H, N, d_B_H, N, dpct::get_value(&beta_H, *handle), d_C_H, N);
  cublasHgemm(handle, trans3, trans3, N, N, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);
}
