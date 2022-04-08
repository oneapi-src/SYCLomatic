// RUN: c2s --format-range=none --usm-level=none -out-root %T/queue_ctn_usm_none %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/queue_ctn_usm_none/queue_ctn_usm_none.dp.cpp


#include "cuda.h"

void bar();
#define SIZE 100

size_t size = 1234567 * sizeof(float);
float *h_A = (float *)malloc(size);
float *d_A = NULL;
__constant__ float constData[1234567 * 4];

// CHECK: void foo1() {
// CHECK-NEXT: c2s::c2s_memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, c2s::device_to_host );
// CHECK-NEXT: c2s::c2s_memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, c2s::device_to_host );
// CHECK-NEXT: c2s::c2s_memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
// CHECK-NEXT: c2s::c2s_memset(d_A, 23, size);
// CHECK-NEXT: c2s::c2s_memset(d_A, 23, size);
// CHECK-NEXT: bar();
// CHECK-NEXT: c2s::c2s_memset(d_A, 23, size);
// CHECK-NEXT: c2s::c2s_memset(d_A, 23, size);
// CHECK-NEXT: }
void foo1() {
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  cudaMemcpyToSymbol(constData, h_A, size, 1);
  cudaMemset(d_A, 23, size);
  cudaMemset(d_A, 23, size);
  bar();
  cudaMemset(d_A, 23, size);
  cudaMemset(d_A, 23, size);
}

