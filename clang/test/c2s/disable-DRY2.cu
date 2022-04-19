// RUN: c2s --format-range=none -out-root %T/disable-DRY2 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/disable-DRY2/disable-DRY2.dp.cpp


#include "cuda.h"

void bar(){
  int device = 0;
// CHECK:/*
// CHECK-NEXT:DPCT1093:{{[0-9]+}}: The "device" may not be the best XPU device. Adjust the selected device if needed.
// CHECK-NEXT:*/
//CHECK-NEXT:c2s::dev_mgr::instance().select_device(device);
  cudaSetDevice(device);
}

#define SIZE 100

size_t size = 1234567 * sizeof(float);
float *h_A = (float *)malloc(size);
float *d_A = NULL;
__constant__ float constData[1234567 * 4];
cudaStream_t s;

// CHECK: void foo1() {
// CHECK-NEXT: c2s::get_default_queue().memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE );
// CHECK-NEXT: c2s::get_default_queue().memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE );
// CHECK-NEXT: c2s::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait();
// CHECK-NEXT: c2s::get_default_queue().memset(d_A, 23, size).wait();
// CHECK-NEXT: c2s::get_default_queue().memset(d_A, 23, size).wait();
// CHECK-NEXT: bar();
// CHECK-NEXT: c2s::get_default_queue().memset(d_A, 23, size).wait();
// CHECK-NEXT: c2s::get_default_queue().memset(d_A, 23, size).wait();
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

