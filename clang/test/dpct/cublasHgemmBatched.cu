// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1
// UNSUPPORTED: v8.0, v9.0, v9.1
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasHgemmBatched.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

cublasHandle_t handle;
int N = 275;

int main() {

  const __half **d_A_H;
  const __half **d_B_H;
  __half **d_C_H;
  __half alpha_H;
  __half beta_H;


  cublasOperation_t trans3 = CUBLAS_OP_N;
  // CHECK: int64_t m_ct{{[0-9]+}} = N, n_ct{{[0-9]+}} = N, k_ct{{[0-9]+}} = N, lda_ct{{[0-9]+}} = N, ldb_ct{{[0-9]+}} = N, ldc_ct{{[0-9]+}} = N, group_size_ct{{[0-9]+}} = 10;
  // CHECK-NEXT: sycl::half alpha_ct{{[0-9]+}} = dpct::get_value(&alpha_H, *handle), beta_ct{{[0-9]+}} = dpct::get_value(&beta_H, *handle);
  // CHECK-NEXT: oneapi::mkl::blas::gemm_batch(*handle, &trans3, &trans3, &m_ct{{[0-9]+}}, &n_ct{{[0-9]+}}, &k_ct{{[0-9]+}}, &alpha_ct{{[0-9]+}}, d_A_H, &lda_ct{{[0-9]+}}, d_B_H, &ldb_ct{{[0-9]+}}, &beta_ct{{[0-9]+}}, d_C_H, &ldc_ct{{[0-9]+}}, 1, &group_size_ct{{[0-9]+}}, {});
  cublasHgemmBatched(handle, trans3, trans3, N, N, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N, 10);
}
