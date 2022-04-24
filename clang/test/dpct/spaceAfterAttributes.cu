// RUN: dpct --format-range=none -out-root %T/spaceAfterAttributes %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck -strict-whitespace %s --match-full-lines --input-file %T/spaceAfterAttributes/spaceAfterAttributes.dp.cpp

#include <cuda_runtime.h>

// CHECK:void foo1() {
void __device__ foo1() {
  return;
}

// CHECK:void foo2() {
void __host__  foo2() {
  return;
}

// CHECK:void foo3() {
void __global__   foo3() {
  return;
}

// CHECK:void foo4() {
void __host__  __device__   foo4() {
  return;
}

// 'constant' attribute only applies to variables,
// here just checking the result after it is removed.
// CHECK:int foo5() {
int __constant__    foo5() {
  return 0;
}

// For checking results, the spaces at the EOL are retained.
// CHECK:#define DEVICE 
#define DEVICE __device__   
#define DEVICE_END

void DEVICE foo6() {
  return;
}

// CHECK:void foo7() {
__host__   void foo7() {
  return;
}


// CHECK:void 
void __host__   
foo8() {
  return;
}

