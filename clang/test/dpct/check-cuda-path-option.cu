// RUN: dpct --format-range=none --cuda-path="%cuda-path" -out-root %T %s  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/check-cuda-path-option.dp.cpp --match-full-lines %s

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CHECK: void foo (int s){
void foo (cublasStatus_t s){
}