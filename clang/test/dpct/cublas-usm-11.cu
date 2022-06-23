// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/cublas-usm-11 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-usm-11/cublas-usm-11.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

void foo1() {
  cublasHandle_t handle;
  void *x, *y, *a, *b, *c, *alpha, *beta, *res, *cos, *sin;
  const void **a_array;
  const void **b_array;
  void **c_array;
  //CHECK:dpct::nrm2(*handle, 4, x, dpct::library_data_t::real_float, 1, res, dpct::library_data_t::real_float);
  //CHECK-NEXT:dpct::dot(*handle, 4, x, dpct::library_data_t::real_float, 1, y, dpct::library_data_t::real_float, 1, res, dpct::library_data_t::real_float);
  //CHECK-NEXT:dpct::dotc(*handle, 4, x, dpct::library_data_t::real_float, 1, y, dpct::library_data_t::real_float, 1, res, dpct::library_data_t::real_float);
  //CHECK-NEXT:dpct::scal(*handle, 4, alpha, dpct::library_data_t::real_float, x, dpct::library_data_t::real_float, 1);
  //CHECK-NEXT:dpct::axpy(*handle, 4, alpha, dpct::library_data_t::real_float, x, dpct::library_data_t::real_float, 1, y, dpct::library_data_t::real_float, 1);
  //CHECK-NEXT:dpct::rot(*handle, 4, x, dpct::library_data_t::real_float, 1, y, dpct::library_data_t::real_float, 1, cos, sin, dpct::library_data_t::real_float);
  //CHECK-NEXT:dpct::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha, a, dpct::library_data_t::real_half, 4, b, dpct::library_data_t::real_half, 4, beta, c, dpct::library_data_t::real_half, 4, dpct::library_data_t::real_half);
  //CHECK-NEXT:dpct::gemm_batch(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha, a_array, dpct::library_data_t::real_half, 4, b_array, dpct::library_data_t::real_half, 4, beta, c_array, dpct::library_data_t::real_half, 4, 2, dpct::library_data_t::real_half);
  //CHECK-NEXT:dpct::gemm_batch(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha, a, dpct::library_data_t::real_half, 4, 16, b, dpct::library_data_t::real_half, 4, 16, beta, c, dpct::library_data_t::real_half, 4, 16, 2, dpct::library_data_t::real_half);
  cublasNrm2Ex(handle, 4, x, CUDA_R_32F, 1, res, CUDA_R_32F, CUDA_R_32F);
  cublasDotEx(handle, 4, x, CUDA_R_32F, 1, y, CUDA_R_32F, 1, res, CUDA_R_32F, CUDA_R_32F);
  cublasDotcEx(handle, 4, x, CUDA_R_32F, 1, y, CUDA_R_32F, 1, res, CUDA_R_32F, CUDA_R_32F);
  cublasScalEx(handle, 4, alpha, CUDA_R_32F, x, CUDA_R_32F, 1, CUDA_R_32F);
  cublasAxpyEx(handle, 4, alpha, CUDA_R_32F, x, CUDA_R_32F, 1, y, CUDA_R_32F, 1, CUDA_R_32F);
  cublasRotEx(handle, 4, x, CUDA_R_32F, 1,  y, CUDA_R_32F, 1,  cos, sin, CUDA_R_32F, CUDA_R_32F);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha, a, CUDA_R_16F, 4, b, CUDA_R_16F, 4, beta, c, CUDA_R_16F, 4, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha, a_array, CUDA_R_16F, 4, b_array, CUDA_R_16F, 4, beta, c_array, CUDA_R_16F, 4, 2, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha, a, CUDA_R_16F, 4, 16, b, CUDA_R_16F, 4, 16, beta, c, CUDA_R_16F, 4, 16, 2, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
}

void foo2() {
  cublasHandle_t handle;
  void *x, *y, *a, *b, *c, *alpha, *beta, *res, *cos, *sin;
  void **a_array;
  void **b_array;
  void **c_array;

  //CHECK:dpct::gemm_batch(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha, (void const **)a_array, dpct::library_data_t::real_half, 4, (void const **)b_array, dpct::library_data_t::real_half, 4, beta, c_array, dpct::library_data_t::real_half, 4, 2, dpct::library_data_t::real_half);
  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha, a_array, CUDA_R_16F, 4, b_array, CUDA_R_16F, 4, beta, c_array, CUDA_R_16F, 4, 2, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
}
