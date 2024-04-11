// RUN: dpct --format-range=none --usm-level=none -out-root %T/cublas-create-Sgemm-destroy %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-create-Sgemm-destroy/cublas-create-Sgemm-destroy.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cublas-create-Sgemm-destroy/cublas-create-Sgemm-destroy.dp.cpp -o %T/cublas-create-Sgemm-destroy/cublas-create-Sgemm-destroy.dp.o %}
// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <cstdio>
// CHECK: #include <dpct/blas_utils.hpp>
// CHECK: #include <dpct/lib_common_utils.hpp>
#include <cstdio>
#include "cublas_v2.h"
#include <cuda_runtime.h>

void foo (cublasStatus_t s){
}
cublasStatus_t bar (cublasStatus_t s){
  return s;
}

// CHECK: extern dpct::blas::descriptor_ptr handle2;
extern cublasHandle_t handle2;

// CHECK: int foo2(dpct::library_data_t DT)  try {
int foo2(cudaDataType DT) {
  // CHECK: int status;
  // CHECK-NEXT: dpct::blas::descriptor_ptr handle;
  // CHECK-NEXT: handle = new dpct::blas::descriptor();
  // CHECK: status = DPCT_CHECK_ERROR(handle = new dpct::blas::descriptor());
  // CHECK-NEXT: if (status != 0) {
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }

  //CHECK: int Atomicsmode;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasGetAtomicsMode was removed because this functionality is redundant in SYCL.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasSetAtomicsMode was removed because this functionality is redundant in SYCL.
  //CHECK-NEXT: */
  cublasAtomicsMode_t Atomicsmode;
  cublasGetAtomicsMode(handle, &Atomicsmode);
  cublasSetAtomicsMode(handle, Atomicsmode);

  //CHECK: int mode = 0;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasGetPointerMode was removed because this functionality is redundant in SYCL.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasSetPointerMode was removed because this functionality is redundant in SYCL.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::library_data_t cdt;
  //CHECK-NEXT: dpct::library_data_t cbdt;
  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
  cublasGetPointerMode(handle, &mode);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  cudaDataType_t cdt;
  cublasDataType_t cbdt;

  // CHECK: dpct::queue_ptr stream1;
  // CHECK-NEXT: stream1 = dpct::get_current_device().create_queue();
  // CHECK-NEXT: handle->set_queue(stream1);
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(handle->set_queue(stream1));
  // CHECK-NEXT: stream1 = &(handle->get_queue());
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(stream1 = &(handle->get_queue()));
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cublasSetStream(handle, stream1);
  status = cublasSetStream(handle, stream1);
  cublasGetStream(handle, &stream1);
  status = cublasGetStream(handle, &stream1);


  int N = 275;
  float *d_A_S = 0;
  float *d_B_S = 0;
  float *d_C_S = 0;
  float alpha_S = 1.0f;
  float beta_S = 0.0f;
  int trans0 = 0;
  int trans1 = 1;
  int trans2 = 2;
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, alpha_S, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_S)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_S)), N, beta_S, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_S)), N, dpct::blas::deduce_compute_mode(std::nullopt, handle->get_math_mode(), false)));
  // CHECK: oneapi::mkl::blas::column_major::gemm(handle->get_queue(), dpct::get_transpose(trans0), dpct::get_transpose(trans1), N, N, N, alpha_S, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_S)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_S)), N, beta_S, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_S)), N, dpct::blas::deduce_compute_mode(std::nullopt, handle->get_math_mode(), false));
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  double *d_A_D = 0;
  double *d_B_D = 0;
  double *d_C_D = 0;
  double alpha_D = 1.0;
  double beta_D = 0.0;
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, alpha_D, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(d_A_D)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(d_B_D)), N, beta_D, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(d_C_D)), N, dpct::blas::deduce_compute_mode(std::nullopt, handle->get_math_mode(), false)));
  // CHECK: oneapi::mkl::blas::column_major::gemm(handle->get_queue(), dpct::get_transpose(trans2), oneapi::mkl::transpose::conjtrans, N, N, N, alpha_D, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(d_A_D)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(d_B_D)), N, beta_D, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(d_C_D)), N, dpct::blas::deduce_compute_mode(std::nullopt, handle->get_math_mode(), false));
  status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  cublasDgemm(handle, (cublasOperation_t)trans2, (cublasOperation_t)2, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);

  __half *d_A_H = 0;
  __half *d_B_H = 0;
  __half *d_C_H = 0;
  __half alpha_H;
  __half beta_H;
  // CHECK: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, alpha_H, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<sycl::half>(d_A_H)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<sycl::half>(d_B_H)), N, beta_H, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<sycl::half>(d_C_H)), N, dpct::blas::deduce_compute_mode(std::nullopt, handle->get_math_mode(), false)));
  // CHECK: oneapi::mkl::blas::column_major::gemm(handle->get_queue(), dpct::get_transpose(trans2), oneapi::mkl::transpose::conjtrans, N, N, N, alpha_H, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<sycl::half>(d_A_H)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<sycl::half>(d_B_H)), N, beta_H, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<sycl::half>(d_C_H)), N, dpct::blas::deduce_compute_mode(std::nullopt, handle->get_math_mode(), false));
  status = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);
  cublasHgemm(handle, (cublasOperation_t)trans2, (cublasOperation_t)2, N, N, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);

  void *alpha, *beta, *A, *B, *C;

  // CHECK: status = DPCT_CHECK_ERROR(dpct::blas::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, alpha, A, dpct::library_data_t::real_half, N, B, dpct::library_data_t::real_half, N, beta, C, dpct::library_data_t::real_half, N, dpct::library_data_t::real_half));
  // CHECK-NEXT: dpct::blas::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, alpha, A, dpct::library_data_t::real_half, N, B, dpct::library_data_t::real_half, N, beta, C, dpct::library_data_t::real_half, N, dpct::library_data_t::real_float);
  // CHECK-NEXT: dpct::blas::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, alpha, A, dpct::library_data_t::real_half, N, B, dpct::library_data_t::real_half, N, beta, C, dpct::library_data_t::real_float, N, dpct::library_data_t::real_float);
  // CHECK-NEXT: dpct::blas::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, alpha, A, dpct::library_data_t::real_float, N, B, dpct::library_data_t::real_float, N, beta, C, dpct::library_data_t::real_float, N, dpct::library_data_t::real_float);
  // CHECK-NEXT: dpct::blas::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, alpha, A, dpct::library_data_t::real_double, N, B, dpct::library_data_t::real_double, N, beta, C, dpct::library_data_t::real_double, N, dpct::library_data_t::real_double);
  // CHECK-NEXT: dpct::blas::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, alpha, A, dpct::library_data_t::complex_float, N, B, dpct::library_data_t::complex_float, N, beta, C, dpct::library_data_t::complex_float, N, dpct::library_data_t::complex_float);
  // CHECK-NEXT: dpct::blas::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, alpha, A, dpct::library_data_t::complex_double, N, B, dpct::library_data_t::complex_double, N, beta, C, dpct::library_data_t::complex_double, N, dpct::library_data_t::complex_double);
  status = cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_16F, N, B, CUDA_R_16F, N, beta, C, CUDA_R_16F, N, CUDA_R_16F, CUBLAS_GEMM_ALGO0);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_16F, N, B, CUDA_R_16F, N, beta, C, CUDA_R_16F, N, CUDA_R_32F, CUBLAS_GEMM_ALGO0);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_16F, N, B, CUDA_R_16F, N, beta, C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_ALGO0);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_32F, N, B, CUDA_R_32F, N, beta, C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_ALGO0);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_64F, N, B, CUDA_R_64F, N, beta, C, CUDA_R_64F, N, CUDA_R_64F, CUBLAS_GEMM_ALGO0);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_C_32F, N, B, CUDA_C_32F, N, beta, C, CUDA_C_32F, N, CUDA_C_32F, CUBLAS_GEMM_ALGO0);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_C_64F, N, B, CUDA_C_64F, N, beta, C, CUDA_C_64F, N, CUDA_C_64F, CUBLAS_GEMM_ALGO0);

  // CHECK: dpct::blas::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, alpha, A, dpct::library_data_t::real_int8, N, B, dpct::library_data_t::real_int8, N, beta, C, dpct::library_data_t::real_int32, N, dpct::library_data_t::real_int32);
  // CHECK-NEXT: dpct::blas::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, alpha, A, DT, N, B, DT, N, beta, C, DT, N, DT);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_8I, N, B, CUDA_R_8I, N, beta, C, CUDA_R_32I, N, CUDA_R_32I, CUBLAS_GEMM_ALGO0);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, DT, N, B, DT, N, beta, C, DT, N, DT, CUBLAS_GEMM_ALGO0);

  float2 alpha_C, beta_C;
  // CHECK: dpct::blas::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, &alpha_S, A, dpct::library_data_t::real_half, N, B, dpct::library_data_t::real_half, N, &beta_S, C, dpct::library_data_t::real_half, N, dpct::library_data_t::real_float);
  // CHECK-NEXT: dpct::blas::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, &alpha_S, A, dpct::library_data_t::real_half, N, B, dpct::library_data_t::real_half, N, &beta_S, C, dpct::library_data_t::real_float, N, dpct::library_data_t::real_float);
  // CHECK-NEXT: dpct::blas::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, &alpha_S, A, dpct::library_data_t::real_float, N, B, dpct::library_data_t::real_float, N, &beta_S, C, dpct::library_data_t::real_float, N, dpct::library_data_t::real_float);
  // CHECK-NEXT: dpct::blas::gemm(handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans, N, N, N, &alpha_C, A, dpct::library_data_t::complex_float, N, B, dpct::library_data_t::complex_float, N, &beta_C, C, dpct::library_data_t::complex_float, N, dpct::library_data_t::complex_float);
  cublasSgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_S, A, CUDA_R_16F, N, B, CUDA_R_16F, N, &beta_S, C, CUDA_R_16F, N);
  cublasSgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_S, A, CUDA_R_16F, N, B, CUDA_R_16F, N, &beta_S, C, CUDA_R_32F, N);
  cublasSgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_S, A, CUDA_R_32F, N, B, CUDA_R_32F, N, &beta_S, C, CUDA_R_32F, N);
  cublasCgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_C, A, CUDA_C_32F, N, B, CUDA_C_32F, N, &beta_C, C, CUDA_C_32F, N);

  // CHECK: for (;;) {
  // CHECK-NEXT: status = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::trans, oneapi::mkl::transpose::trans, N, N, N, alpha_S, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_S)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_S)), N, beta_S, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_S)), N, dpct::blas::deduce_compute_mode(std::nullopt, handle->get_math_mode(), false)));
  // CHECK-NEXT: beta_S = beta_S + 1;
  // CHECK-NEXT: }
  // CHECK-NEXT: alpha_S = alpha_S + 1;
  for (;;) {
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
    beta_S = beta_S + 1;
  }
  alpha_S = alpha_S + 1;

  // CHECK: for (;;) {
  // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::trans, oneapi::mkl::transpose::trans, N, N, N, alpha_S, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_S)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_S)), N, beta_S, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_S)), N, dpct::blas::deduce_compute_mode(std::nullopt, handle->get_math_mode(), false));
  // CHECK-NEXT: beta_S = beta_S + 1;
  // CHECK-NEXT: }
  // CHECK-NEXT: alpha_S = alpha_S + 1;
  for (;;) {
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
    beta_S = beta_S + 1;
  }
  alpha_S = alpha_S + 1;


  // CHECK: foo(bar(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, alpha_S, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_A_S)), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_S)), N, beta_S, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_S)), N, dpct::blas::deduce_compute_mode(std::nullopt, handle->get_math_mode(), false)))));
  foo(bar(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)));

#define dA(i, j) *(d_A_S + (i) + (j) * N)
  // CHECK: oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, alpha_S, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(&dA(10, 20))), N, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_B_S)), N, beta_S, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(d_C_S)), N, dpct::blas::deduce_compute_mode(std::nullopt, handle->get_math_mode(), false));
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, &dA(10, 20), N, d_B_S, N, &beta_S, d_C_S, N);
#undef dA(i, j)

  // CHECK: status = DPCT_CHECK_ERROR(delete (handle));
  // CHECK-NEXT: delete (handle);
  // CHECK-NEXT: return 0;
  status = cublasDestroy(handle);
  cublasDestroy(handle);
  return 0;
}

void foo3(cublasHandle_t handle) {
  int ver;
  // CHECK: int err = DPCT_CHECK_ERROR(dpct::mkl_get_version(dpct::version_field::major, &ver));
  int err = cublasGetVersion(handle, &ver);
}

void foo4() {
  cublasHandle_t handle;
  float   *a_f, *b_f, *x_f, *c_f, *alpha_f, *beta_f;
  double  *a_d, *b_d, *x_d, *c_d, *alpha_d, *beta_d;
  float2  *a_c, *b_c, *x_c, *c_c, *alpha_c, *beta_c;
  double2 *a_z, *b_z, *x_z, *c_z, *alpha_z, *beta_z;

  //CHECK:oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), oneapi::mkl::side::left, 2, 2, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(a_f)), 2, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_f)), 1, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(c_f)), 2);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), oneapi::mkl::side::left, 2, 2, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(a_d)), 2, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(x_d)), 1, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<double>(c_d)), 2);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), oneapi::mkl::side::left, 2, 2, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(a_c)), 2, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(x_c)), 1, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<float>>(c_c)), 2);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), oneapi::mkl::side::left, 2, 2, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(a_z)), 2, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(x_z)), 1, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::complex<double>>(c_z)), 2);
  cublasSdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a_f, 2, x_f, 1, c_f, 2);
  cublasDdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a_d, 2, x_d, 1, c_d, 2);
  cublasCdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a_c, 2, x_c, 1, c_c, 2);
  cublasZdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a_z, 2, x_z, 1, c_z, 2);
}

void foo() {
  //CHECK:const dpct::blas::descriptor_ptr h_c = nullptr;
  //CHECK-NEXT:dpct::blas::descriptor_ptr h = h_c;
  const cublasHandle_t h_c = nullptr;
  cublasHandle_t h = h_c;
}

#define CHECK_INTERNAL(err)                                                    \
  { auto err_ = (err); }

#define CHECK(err) CHECK_INTERNAL(err)

void foo2(cublasHandle_t *handles, int i, cudaStream_t s) {
  // CHECK: CHECK(DPCT_CHECK_ERROR(handles[i] = new dpct::blas::descriptor()));
  // CHECK-NEXT: CHECK(DPCT_CHECK_ERROR(delete (handles[i])));
  // CHECK-NEXT: CHECK(DPCT_CHECK_ERROR(handles[i]->set_queue(s)));
  // CHECK-NEXT: CHECK(DPCT_CHECK_ERROR(s = &(handles[i]->get_queue())));
  CHECK(cublasCreate(&handles[i]));
  CHECK(cublasDestroy(handles[i]));
  CHECK(cublasSetStream(handles[i], s));
  CHECK(cublasGetStream(handles[i], &s));
}
