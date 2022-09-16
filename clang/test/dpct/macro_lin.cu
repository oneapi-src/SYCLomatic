// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none -out-root %T/macro_lin %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/macro_lin/macro_lin.dp.cpp

#include "cuda.h"
#include <cstdio>

// CHECK: #define AAA
#define AAA CUDART_CB

// CHECK: void my_callback(dpct::queue_ptr stream, int status, void *data) {
// CHECK-NEXT:   printf("callback from stream %d\n", *((int *)data));
// CHECK-NEXT: }
void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void *data) {
  printf("callback from stream %d\n", *((int *)data));
}

