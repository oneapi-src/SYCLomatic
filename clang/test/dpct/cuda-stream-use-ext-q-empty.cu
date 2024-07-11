// RUN: dpct --usm-level=none  -out-root %T/cuda-stream-use-ext-q-empty %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cuda-stream-use-ext-q-empty/cuda-stream-use-ext-q-empty.dp.cpp --match-full-lines %s

#include <cuda_runtime.h>
#include <iostream>

template <typename T>
void my_error_checker(T ReturnValue, char const *const FuncName) {
}

#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)

template <typename FloatN, typename Float>
static void func() {
  cudaStream_t s0;
  
  // CHECK:  s0->ext_oneapi_empty();
  // CHECK-NEXT:  MY_ERROR_CHECKER(DPCT_CHECK_ERROR((s0->ext_oneapi_empty())));
  // CHECK-NEXT:  dpct::err0 status = cudaStreamQuery(stream);
  // CHECK-NEXT:  status = DPCT_CHECK_ERROR((q_ct1.ext_oneapi_empty()));
  cudaStreamQuery(s0);
  MY_ERROR_CHECKER(cudaStreamQuery(s0));
  cudaError_t status = cudaStreamQuery(stream);
  status = cudaStreamQuery(0);
}
