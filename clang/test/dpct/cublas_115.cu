// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4
// RUN: dpct --format-range=none -out-root %T/cublas_115 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas_115/cublas_115.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cublas_115/cublas_115.dp.cpp -o %T/cublas_115/cublas_115.dp.o %}

#include <cstdio>
#include <cublas_v2.h>

void foo1(cublasStatus_t s) {
  //CHECK:printf("Error string: %s", dpct::get_error_dummy(s));
  printf("Error string: %s", cublasGetStatusString(s));
  cublasHandle_t handle;
  void *workspace;
  size_t size;
  //CHECK:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to cublasSetWorkspace was removed because this functionality is redundant in SYCL.
  //CHECK-NEXT:*/
  cublasSetWorkspace(handle, workspace, size);
}

//CHECK:void foo2(dpct::compute_type &a) {
//CHECK-NEXT:  a = dpct::compute_type::f16;
//CHECK-NEXT:  a = dpct::compute_type::f16_standard;
//CHECK-NEXT:  a = dpct::compute_type::f32;
//CHECK-NEXT:  a = dpct::compute_type::f32_standard;
//CHECK-NEXT:  a = dpct::compute_type::f32;
//CHECK-NEXT:  a = dpct::compute_type::f32_fast_bf16;
//CHECK-NEXT:  a = dpct::compute_type::f32_fast_tf32;
//CHECK-NEXT:  a = dpct::compute_type::f64;
//CHECK-NEXT:  a = dpct::compute_type::f64_standard;
//CHECK-NEXT:  a = dpct::compute_type::i32;
//CHECK-NEXT:  a = dpct::compute_type::i32_standard;
//CHECK-NEXT:}
void foo2(cublasComputeType_t &a) {
  a = CUBLAS_COMPUTE_16F;
  a = CUBLAS_COMPUTE_16F_PEDANTIC;
  a = CUBLAS_COMPUTE_32F;
  a = CUBLAS_COMPUTE_32F_PEDANTIC;
  a = CUBLAS_COMPUTE_32F_FAST_16F;
  a = CUBLAS_COMPUTE_32F_FAST_16BF;
  a = CUBLAS_COMPUTE_32F_FAST_TF32;
  a = CUBLAS_COMPUTE_64F;
  a = CUBLAS_COMPUTE_64F_PEDANTIC;
  a = CUBLAS_COMPUTE_32I;
  a = CUBLAS_COMPUTE_32I_PEDANTIC;
}
