// RUN: dpct -out-root %T %s --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-lambda.dp.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <cstdio>
// CHECK: #include <mkl_blas_sycl.hpp>
// CHECK-NEXT: #include <mkl_lapack_sycl.hpp>
// CHECK-NEXT: #include <mkl_sycl_types.hpp>
#include <cstdio>
#include "cublas_v2.h"
#include <cuda_runtime.h>


int main() {
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);

  int N = 275;
  float *d_A_S = 0;
  float *d_B_S = 0;
  float *d_C_S = 0;
  float alpha_S = 1.0f;
  float beta_S = 0.0f;


  // CHECK: if ([&]() {
  // CHECK-NEXT: auto allocation_ct7 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(
  // CHECK-NEXT:     cl::sycl::range<1>(allocation_ct7.size / sizeof(float)));
  // CHECK-NEXT: auto allocation_ct9 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct9 = allocation_ct9.buffer.reinterpret<float>(
  // CHECK-NEXT:     cl::sycl::range<1>(allocation_ct9.size / sizeof(float)));
  // CHECK-NEXT: auto allocation_ct12 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct12 =
  // CHECK-NEXT:     allocation_ct12.buffer.reinterpret<float>(
  // CHECK-NEXT:         cl::sycl::range<1>(allocation_ct12.size / sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N,
  // CHECK-NEXT:                 N, N, *(&alpha_S), buffer_ct7, N, buffer_ct9, N, *(&beta_S),
  // CHECK-NEXT:                 buffer_ct12, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT:     }()) {
  // CHECK-NEXT: }
  if(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){
  }


  // CHECK: if (int stat = [&]() {
  // CHECK-NEXT: auto allocation_ct7 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(
  // CHECK-NEXT:     cl::sycl::range<1>(allocation_ct7.size / sizeof(float)));
  // CHECK-NEXT: auto allocation_ct9 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct9 = allocation_ct9.buffer.reinterpret<float>(
  // CHECK-NEXT:     cl::sycl::range<1>(allocation_ct9.size / sizeof(float)));
  // CHECK-NEXT: auto allocation_ct12 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct12 =
  // CHECK-NEXT:     allocation_ct12.buffer.reinterpret<float>(
  // CHECK-NEXT:         cl::sycl::range<1>(allocation_ct12.size / sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N,
  // CHECK-NEXT:                 N, N, *(&alpha_S), buffer_ct7, N, buffer_ct9, N, *(&beta_S),
  // CHECK-NEXT:                 buffer_ct12, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT:     }()) {
  // CHECK-NEXT: }
  if(int stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){
  }


  // CHECK: for ([&]() {
  // CHECK-NEXT: auto allocation_ct7 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(
  // CHECK-NEXT:     cl::sycl::range<1>(allocation_ct7.size / sizeof(float)));
  // CHECK-NEXT: auto allocation_ct9 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct9 = allocation_ct9.buffer.reinterpret<float>(
  // CHECK-NEXT:     cl::sycl::range<1>(allocation_ct9.size / sizeof(float)));
  // CHECK-NEXT: auto allocation_ct12 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct12 =
  // CHECK-NEXT:     allocation_ct12.buffer.reinterpret<float>(
  // CHECK-NEXT:         cl::sycl::range<1>(allocation_ct12.size / sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N,
  // CHECK-NEXT:                 N, N, *(&alpha_S), buffer_ct7, N, buffer_ct9, N, *(&beta_S),
  // CHECK-NEXT:                 buffer_ct12, N);
  // CHECK-NEXT:      }();
  // CHECK-NEXT:      ;) {
  // CHECK-NEXT: }
  for(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);;){
  }


  // CHECK: while ([&]() {
  // CHECK-NEXT: auto allocation_ct7 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(
  // CHECK-NEXT:     cl::sycl::range<1>(allocation_ct7.size / sizeof(float)));
  // CHECK-NEXT: auto allocation_ct9 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct9 = allocation_ct9.buffer.reinterpret<float>(
  // CHECK-NEXT:     cl::sycl::range<1>(allocation_ct9.size / sizeof(float)));
  // CHECK-NEXT: auto allocation_ct12 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct12 =
  // CHECK-NEXT:     allocation_ct12.buffer.reinterpret<float>(
  // CHECK-NEXT:         cl::sycl::range<1>(allocation_ct12.size / sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N,
  // CHECK-NEXT:                 N, N, *(&alpha_S), buffer_ct7, N, buffer_ct9, N, *(&beta_S),
  // CHECK-NEXT:                 buffer_ct12, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }() != 0) {
  // CHECK-NEXT: }
  while(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)!=0){
  }


  // CHECK: do{
  // CHECK-NEXT: } while ([&]() {
  // CHECK-NEXT: auto allocation_ct7 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(
  // CHECK-NEXT:     cl::sycl::range<1>(allocation_ct7.size / sizeof(float)));
  // CHECK-NEXT: auto allocation_ct9 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct9 = allocation_ct9.buffer.reinterpret<float>(
  // CHECK-NEXT:     cl::sycl::range<1>(allocation_ct9.size / sizeof(float)));
  // CHECK-NEXT: auto allocation_ct12 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct12 =
  // CHECK-NEXT:     allocation_ct12.buffer.reinterpret<float>(
  // CHECK-NEXT:         cl::sycl::range<1>(allocation_ct12.size / sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N,
  // CHECK-NEXT:                 N, N, *(&alpha_S), buffer_ct7, N, buffer_ct9, N, *(&beta_S),
  // CHECK-NEXT:                 buffer_ct12, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }());
  do{
  }while(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N));


  // CHECK: switch (int stat = [&]() {
  // CHECK-NEXT: auto allocation_ct7 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(
  // CHECK-NEXT:     cl::sycl::range<1>(allocation_ct7.size / sizeof(float)));
  // CHECK-NEXT: auto allocation_ct9 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct9 = allocation_ct9.buffer.reinterpret<float>(
  // CHECK-NEXT:     cl::sycl::range<1>(allocation_ct9.size / sizeof(float)));
  // CHECK-NEXT: auto allocation_ct12 =
  // CHECK-NEXT:     dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct12 =
  // CHECK-NEXT:     allocation_ct12.buffer.reinterpret<float>(
  // CHECK-NEXT:         cl::sycl::range<1>(allocation_ct12.size / sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N,
  // CHECK-NEXT:                 N, N, *(&alpha_S), buffer_ct7, N, buffer_ct9, N, *(&beta_S),
  // CHECK-NEXT:                 buffer_ct12, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }()) {}
  switch (int stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){
  }


  return 0;
}
