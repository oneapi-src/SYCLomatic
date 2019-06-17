// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/cublas-create-Sgemm-destroy.sycl.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <syclct/syclct.hpp>
// CHECK-NEXT: #include <cstdio>
// CHECK: #include <mkl_blas_sycl.hpp>
// CHECK-NEXT: #include <mkl_lapack_sycl.hpp>
// CHECK-NEXT: #include <sycl_types.hpp>
// CHECK: #include <complex>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

void foo (cublasStatus_t s){
}
cublasStatus_t bar (cublasStatus_t s){
  return s;
}

int main() {
  // CHECK: int status;
  // CHECK-NEXT: cl::sycl::queue handle;
  // CHECK-NEXT: status = 0;
  // CHECK-NEXT: if (status != 0) {
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }
  int N = 275;
  float *d_A_S = 0;
  float *d_B_S = 0;
  float *d_C_S = 0;
  float alpha_S = 1.0f;
  float beta_S = 0.0f;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::sgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_S), d_A_S_BUFFER_{{[0-9,a-z]+}}, N, d_B_S_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_S), d_C_S_BUFFER_{{[0-9,a-z]+}}, N), 0);
  // CHECK: mkl::sgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_S), d_A_S_BUFFER_{{[0-9,a-z]+}}, N, d_B_S_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_S), d_C_S_BUFFER_{{[0-9,a-z]+}}, N);
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  double *d_A_D = 0;
  double *d_B_D = 0;
  double *d_C_D = 0;
  double alpha_D = 1.0;
  double beta_D = 0.0;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_D), d_A_D_BUFFER_{{[0-9,a-z]+}}, N, d_B_D_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_D), d_C_D_BUFFER_{{[0-9,a-z]+}}, N), 0);
  // CHECK: mkl::dgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_D), d_A_D_BUFFER_{{[0-9,a-z]+}}, N, d_B_D_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_D), d_C_D_BUFFER_{{[0-9,a-z]+}}, N);
  status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  float2 *d_A_C = 0;
  float2 *d_B_C = 0;
  float2 *d_C_C = 0;
  float2 alpha_C = make_float2(1, 0);
  float2 beta_C = make_float2(0, 0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::cgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_C), d_A_C_BUFFER_{{[0-9,a-z]+}}, N, d_B_C_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_C), d_C_C_BUFFER_{{[0-9,a-z]+}}, N), 0);
  // CHECK: mkl::cgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_C), d_A_C_BUFFER_{{[0-9,a-z]+}}, N, d_B_C_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_C), d_C_C_BUFFER_{{[0-9,a-z]+}}, N);
  status = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N);
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N);
  double2 *d_A_Z = 0;
  double2 *d_B_Z = 0;
  double2 *d_C_Z = 0;
  double2 alpha_Z = make_double2(1, 0);
  double2 beta_Z = make_double2(0, 0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::zgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_Z), d_A_Z_BUFFER_{{[0-9,a-z]+}}, N, d_B_Z_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_Z), d_C_Z_BUFFER_{{[0-9,a-z]+}}, N), 0);
  // CHECK: mkl::zgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_Z), d_A_Z_BUFFER_{{[0-9,a-z]+}}, N, d_B_Z_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_Z), d_C_Z_BUFFER_{{[0-9,a-z]+}}, N);
  status = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, &beta_Z, d_C_Z, N);
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, &beta_Z, d_C_Z, N);
  cuComplex *d_A_C_2 = 0;
  cuComplex *d_B_C_2 = 0;
  cuComplex *d_C_C_2 = 0;
  cuComplex alpha_C_2 = make_float2(1, 0);
  cuComplex beta_C_2 = make_float2(0, 0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::cgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_C_2), d_A_C_2_BUFFER_{{[0-9,a-z]+}}, N, d_B_C_2_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_C_2), d_C_C_2_BUFFER_{{[0-9,a-z]+}}, N), 0);
  // CHECK: mkl::cgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_C_2), d_A_C_2_BUFFER_{{[0-9,a-z]+}}, N, d_B_C_2_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_C_2), d_C_C_2_BUFFER_{{[0-9,a-z]+}}, N);
  status = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C_2, d_A_C_2, N, d_B_C_2, N, &beta_C_2, d_C_C_2, N);
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C_2, d_A_C_2, N, d_B_C_2, N, &beta_C_2, d_C_C_2, N);
  cuDoubleComplex *d_A_Z_2 = 0;
  cuDoubleComplex *d_B_Z_2 = 0;
  cuDoubleComplex *d_C_Z_2 = 0;
  cuDoubleComplex alpha_Z_2 = make_double2(1, 0);
  cuDoubleComplex beta_Z_2 = make_double2(0, 0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::zgemm(handle, mkl::transpose::trans, mkl::transpose::trans, N, N, N, *(&alpha_Z_2), d_A_Z_2_BUFFER_{{[0-9,a-z]+}}, N, d_B_Z_2_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_Z_2), d_C_Z_2_BUFFER_{{[0-9,a-z]+}}, N), 0);
  // CHECK: mkl::zgemm(handle, mkl::transpose::conjtrans, mkl::transpose::conjtrans, N, N, N, *(&alpha_Z_2), d_A_Z_2_BUFFER_{{[0-9,a-z]+}}, N, d_B_Z_2_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_Z_2), d_C_Z_2_BUFFER_{{[0-9,a-z]+}}, N);
  status = cublasZgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha_Z_2, d_A_Z_2, N, d_B_Z_2, N, &beta_Z_2, d_C_Z_2, N);
  cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_Z_2, d_A_Z_2, N, d_B_Z_2, N, &beta_Z_2, d_C_Z_2, N);


  // CHECK: for (;;) {
  // CHECK-NEXT: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto d_A_S_ALLOCATION_{{[0-9,a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> d_A_S_BUFFER_{{[0-9,a-z]+}} = d_A_S_ALLOCATION_{{[0-9,a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(d_A_S_ALLOCATION_{{[0-9,a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto d_B_S_ALLOCATION_{{[0-9,a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> d_B_S_BUFFER_{{[0-9,a-z]+}} = d_B_S_ALLOCATION_{{[0-9,a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(d_B_S_ALLOCATION_{{[0-9,a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto d_C_S_ALLOCATION_{{[0-9,a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> d_C_S_BUFFER_{{[0-9,a-z]+}} = d_C_S_ALLOCATION_{{[0-9,a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(d_C_S_ALLOCATION_{{[0-9,a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::sgemm(handle, mkl::transpose::trans, mkl::transpose::trans, N, N, N, *(&alpha_S), d_A_S_BUFFER_{{[0-9,a-z]+}}, N, d_B_S_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_S), d_C_S_BUFFER_{{[0-9,a-z]+}}, N), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: beta_S = beta_S + 1;
  // CHECK-NEXT: }
  // CHECK-NEXT: alpha_S = alpha_S + 1;
  for (;;) {
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
    beta_S = beta_S + 1;
  }
  alpha_S = alpha_S + 1;

  // CHECK: for (;;) {
  // CHECK-NEXT: {
  // CHECK-NEXT: auto d_A_S_ALLOCATION_{{[0-9,a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> d_A_S_BUFFER_{{[0-9,a-z]+}} = d_A_S_ALLOCATION_{{[0-9,a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(d_A_S_ALLOCATION_{{[0-9,a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto d_B_S_ALLOCATION_{{[0-9,a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> d_B_S_BUFFER_{{[0-9,a-z]+}} = d_B_S_ALLOCATION_{{[0-9,a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(d_B_S_ALLOCATION_{{[0-9,a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto d_C_S_ALLOCATION_{{[0-9,a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> d_C_S_BUFFER_{{[0-9,a-z]+}} = d_C_S_ALLOCATION_{{[0-9,a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(d_C_S_ALLOCATION_{{[0-9,a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::sgemm(handle, mkl::transpose::trans, mkl::transpose::trans, N, N, N, *(&alpha_S), d_A_S_BUFFER_{{[0-9,a-z]+}}, N, d_B_S_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_S), d_C_S_BUFFER_{{[0-9,a-z]+}}, N);
  // CHECK-NEXT: }
  // CHECK-NEXT: beta_S = beta_S + 1;
  // CHECK-NEXT: }
  // CHECK-NEXT: alpha_S = alpha_S + 1;
  for (;;) {
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
    beta_S = beta_S + 1;
  }
  alpha_S = alpha_S + 1;


  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto d_A_S_ALLOCATION_{{[0-9,a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> d_A_S_BUFFER_{{[0-9,a-z]+}} = d_A_S_ALLOCATION_{{[0-9,a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(d_A_S_ALLOCATION_{{[0-9,a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto d_B_S_ALLOCATION_{{[0-9,a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> d_B_S_BUFFER_{{[0-9,a-z]+}} = d_B_S_ALLOCATION_{{[0-9,a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(d_B_S_ALLOCATION_{{[0-9,a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto d_C_S_ALLOCATION_{{[0-9,a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> d_C_S_BUFFER_{{[0-9,a-z]+}} = d_C_S_ALLOCATION_{{[0-9,a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(d_C_S_ALLOCATION_{{[0-9,a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: foo(bar((mkl::sgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_S), d_A_S_BUFFER_{{[0-9,a-z]+}}, N, d_B_S_BUFFER_{{[0-9,a-z]+}}, N, *(&beta_S), d_C_S_BUFFER_{{[0-9,a-z]+}}, N), 0)));
  foo(bar(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)));

  // CHECK: status = 0;
  // CHECK-NEXT: return 0;
  status = cublasDestroy(handle);
  cublasDestroy(handle);
  return 0;
}