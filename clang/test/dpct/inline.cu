// RUN: dpct --format-range=none -out-root %T/inline %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/inline/inline.dp.cpp
#include <cuda_runtime.h>

#define NUM_ELEMENTS 16

__device__ float out[NUM_ELEMENTS];

// CHECK: __dpct_inline__ void kernel1(sycl::nd_item<3> [[ITEM:item_ct1]], float *out) {
// CHECK:   out[{{.*}}[[ITEM]].get_local_id(2)] = [[ITEM]].get_local_id(2);
// CHECK: }
__forceinline__ __global__ void kernel1() {
  out[threadIdx.x] = threadIdx.x;
}

// CHECK: #define NEW_INLINE __dpct_inline__
#define NEW_INLINE __forceinline__

// CHECK: NEW_INLINE  void kernel2() {
NEW_INLINE __global__ void kernel2() {
  int a = 2;
}

// CHECK: #define INLINE_KERNEL3 __dpct_inline__  void kernel3() {int a = 2;}
#define INLINE_KERNEL3 __forceinline__ __global__ void kernel3() {int a = 2;}

INLINE_KERNEL3

class TestClass {
  TestClass();
  // CHECK: __dpct_inline__ void foo(){
  __forceinline__ void foo(){
    int a = 2;
  };
  template <class T>
  // CHECK: NEW_INLINE T foo2(T in){
  NEW_INLINE T foo2(T in){
    return in;
  };
};

// CHECK: extern __dpct_inline__ void error(void);
extern __forceinline__ void error(void);

