// UNSUPPORTED: cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-11.9, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6
// UNSUPPORTED: v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v11.9, v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6
// RUN: dpct --format-range=none --usm-level=none -out-root %T/nestedqueue %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/nestedqueue/nestedqueue.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/nestedqueue/nestedqueue.dp.cpp -o %T/nestedqueue/nestedqueue.dp.o %}

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__host__ __device__ void foo1(){
  // CHECK: /*
  // CHECK-NEXT: DPCT1021:{{[0-9]+}}: Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasHandle_t handle;
  cublasHandle_t handle;
  int n=1;
  float* x_S=0;
  int incx=1;
  int* result =0;
  // CHECK: /*
  // CHECK-NEXT: DPCT1020:{{[0-9]+}}: Migration of cublasIsamax, if it is called from __global__ or __device__ function, is not supported. You may need to redesign the code to use the host-side oneapi::mkl::blas::column_major::iamax instead, which submits this call to the SYCL queue automatically.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);
}

__device__ void foo2(){
  // CHECK: /*
  // CHECK-NEXT: DPCT1021:{{[0-9]+}}: Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasHandle_t handle;
  cublasHandle_t handle;
  int n=1;
  float* x_S=0;
  int incx=1;
  int* result =0;
  // CHECK: /*
  // CHECK-NEXT: DPCT1020:{{[0-9]+}}: Migration of cublasIsamax, if it is called from __global__ or __device__ function, is not supported. You may need to redesign the code to use the host-side oneapi::mkl::blas::column_major::iamax instead, which submits this call to the SYCL queue automatically.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);
}

__global__ void foo3(){
  // CHECK: /*
  // CHECK-NEXT: DPCT1021:{{[0-9]+}}: Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasHandle_t handle;
  cublasHandle_t handle;
  int n=1;
  float* x_S=0;
  int incx=1;
  int* result =0;
  // CHECK: /*
  // CHECK-NEXT: DPCT1020:{{[0-9]+}}: Migration of cublasIsamax, if it is called from __global__ or __device__ function, is not supported. You may need to redesign the code to use the host-side oneapi::mkl::blas::column_major::iamax instead, which submits this call to the SYCL queue automatically.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);
}

__host__ void foo4(){
  // CHECK: dpct::blas::descriptor_ptr handle;
  cublasHandle_t handle;
  int n=1;
  float* x_S=0;
  int incx=1;
  int* result =0;
  // CHECK: [&]() {
  // CHECK-NEXT: dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), result);
  // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x_S)), incx, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<std::int64_t>(res_wrapper_ct4.get_ptr())), oneapi::mkl::index_base::one);
  // CHECK-NEXT: return 0;
  // CHECK-NEXT: }();
  cublasIsamax(handle, n, x_S, incx, result);
}

__global__ void childKernel() {}
__global__ void parentKernel() {
  // CHECK: /*
  // CHECK-NEXT: DPCT1130:{{[0-9]+}}: SYCL 2020 standard does not support dynamic parallelism (launching kernel in device code). Please rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: childKernel<<<1, 1>>>();
  childKernel<<<1, 1>>>();
}
void foo5() {
  // CHECK: dpct::get_out_of_order_queue().parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:     parentKernel();
  // CHECK-NEXT:   });
  parentKernel<<<1, 1>>>();
}
