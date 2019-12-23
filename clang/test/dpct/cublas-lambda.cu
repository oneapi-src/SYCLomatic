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


  // CHECK: if([&](){
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_S), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, *(&beta_S), buffer_ct{{[0-9]+}}, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }()){
  // CHECK-NEXT: }
  if(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){
  }


  // CHECK: if(int stat = [&](){
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_S), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, *(&beta_S), buffer_ct{{[0-9]+}}, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }()){
  // CHECK-NEXT: }
  if(int stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){
  }


  // CHECK: for([&](){
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_S), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, *(&beta_S), buffer_ct{{[0-9]+}}, N);
  // CHECK-NEXT: }();;){
  // CHECK-NEXT: }
  for(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);;){
  }


  // CHECK: while([&](){
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_S), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, *(&beta_S), buffer_ct{{[0-9]+}}, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }()!=0){
  // CHECK-NEXT: }
  while(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)!=0){
  }


  // CHECK: do{
  // CHECK-NEXT: }while([&](){
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_S), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, *(&beta_S), buffer_ct{{[0-9]+}}, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }());
  do{
  }while(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N));


  // CHECK: switch (int stat = [&](){
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_S), buffer_ct{{[0-9]+}}, N, buffer_ct{{[0-9]+}}, N, *(&beta_S), buffer_ct{{[0-9]+}}, N);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }()){
  // CHECK-NEXT: }
  switch (int stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){
  }


  return 0;
}
