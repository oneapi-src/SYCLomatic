// RUN: dpct --format-range=none --cuda-include-path="%cuda-path/include" -out-root %T/check-cuda-include-path-option %s -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/check-cuda-include-path-option/check-cuda-include-path-option.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/check-cuda-include-path-option/check-cuda-include-path-option.dp.cpp -o %T/check-cuda-include-path-option/check-cuda-include-path-option.dp.o %}

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CHECK: void foo (int s){
void foo (cublasStatus_t s){
}

