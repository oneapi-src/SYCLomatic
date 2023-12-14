// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1
// UNSUPPORTED: v8.0, v9.0, v9.1
// RUN: dpct --format-range=none -out-root %T/cublasHgemmBatched %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasHgemmBatched/cublasHgemmBatched.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cublasHgemmBatched/cublasHgemmBatched.dp.cpp -o %T/cublasHgemmBatched/cublasHgemmBatched.dp.o %}
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
  // CHECK: dpct::gemm_batch(handle->get_queue(), trans3, trans3, N, N, N, &alpha_H, (const void**)d_A_H, dpct::library_data_t::real_half, N, (const void**)d_B_H, dpct::library_data_t::real_half, N, &beta_H, (void**)d_C_H, dpct::library_data_t::real_half, N, 10, dpct::library_data_t::real_half);
  cublasHgemmBatched(handle, trans3, trans3, N, N, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N, 10);
}

