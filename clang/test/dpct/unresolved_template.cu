// RUN: dpct --format-range=none -out-root %T/unresolved_template %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/unresolved_template/unresolved_template.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/unresolved_template/unresolved_template.dp.cpp -o %T/unresolved_template/unresolved_template.dp.o %}

#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef __HIP_PLATFORM_HCC__
#include <mma.h>
#endif
#include <stdio.h>

template <typename T>
int CUBLAS_GEMM_EX(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float *alpha,
                   const float *beta,
                   const T *A,
                   const T *B,
                   T *C,
                   cublasGemmAlgo_t algo) {

  constexpr auto cublas_dtype_16 = std::is_same<T, __half>::value ? CUDA_R_16F : CUDA_R_16BF;
  // CHECK: int status = DPCT_CHECK_ERROR(dpct::gemm(handle->get_queue(), transa, transb, m, n, k, (const void *)alpha, (const void *)A, cublas_dtype_16, (transa == oneapi::mkl::transpose::nontrans) ? m : k, (const void *)B, cublas_dtype_16, (transb == oneapi::mkl::transpose::nontrans) ? k : n, (const void *)beta, (void *)C, cublas_dtype_16, m, dpct::library_data_t::real_float));
  cublasStatus_t status = cublasGemmEx(handle,
                                       transa,
                                       transb,
                                       m,
                                       n,
                                       k,
                                       (const void *)alpha,
                                       (const void *)A,
                                       cublas_dtype_16,
                                       (transa == CUBLAS_OP_N) ? m : k,
                                       (const void *)B,
                                       cublas_dtype_16,
                                       (transb == CUBLAS_OP_N) ? k : n,
                                       (const void *)beta,
                                       (void *)C,
                                       cublas_dtype_16,
                                       m,
                                       CUDA_R_32F,
                                       algo);
}
