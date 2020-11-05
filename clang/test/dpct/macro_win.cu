// UNSUPPORTED: -linux-
// RUN: dpct --format-range=none -out-root %T/macro_win %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/macro_win/macro_win.dp.cpp

#include "cuda.h"
#include <cstdio>

// CHECK: #define AAA __stdcall
#define AAA CUDART_CB

// CHECK: void __stdcall my_callback(sycl::queue *stream, int status, void *data) {
// CHECK-NEXT:   printf("callback from stream %d\n", *((int *)data));
// CHECK-NEXT: }
void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void *data) {
  printf("callback from stream %d\n", *((int *)data));
}

