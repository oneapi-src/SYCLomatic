// RUN: dpct --format-range=none -out-root %T/noinline %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/noinline/noinline.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/noinline/noinline.dp.cpp -o %T/noinline/noinline.dp.o %}
#include <cuda_runtime.h>

#define NUM_ELEMENTS 16

__device__ float out[NUM_ELEMENTS];

// CHECK: __dpct_noinline__ void kernel1(const sycl::nd_item<3> &[[ITEM:item_ct1]], float *out) {
// CHECK:   out[{{.*}}[[ITEM]].get_local_id(2)] = [[ITEM]].get_local_id(2);
// CHECK: }
__noinline__ __global__ void kernel1() {
  out[threadIdx.x] = threadIdx.x;
}

// CHECK: #define NO_INLINE __dpct_noinline__
#define NO_INLINE __noinline__

// CHECK: NO_INLINE  void kernel2() {
NO_INLINE __global__ void kernel2() {
  int a = 2;
}

// CHECK: #define NO_INLINE_KERNEL __dpct_noinline__  void kernel3() {int a = 2;}
#define NO_INLINE_KERNEL __noinline__ __global__ void kernel3() {int a = 2;}

NO_INLINE_KERNEL

class TestClass {
  TestClass();
  // CHECK: __dpct_noinline__ void foo(){
  __noinline__ void foo(){
    int a = 2;
  };
  template <class T>
  // CHECK: NO_INLINE T foo2(T in){
  NO_INLINE T foo2(T in){
    return in;
  };
};

// CHECK: extern __dpct_noinline__ void error(void);
extern __noinline__ void error(void);

template <typename scalar_t>
// CHECK: __dpct_noinline__ scalar_t calc_igammac(scalar_t a, scalar_t b) {
__noinline__ __host__ __device__ scalar_t calc_igammac(scalar_t a, scalar_t b) {
  scalar_t c = a + b;
  return c;
}

// CHECK: __attribute__((__noinline__)) void macro_with_attr() {}
__attribute__((__noinline__)) void macro_with_attr() {}
// CHECK: #define NOINLINE __attribute__((__noinline__))
#define NOINLINE __attribute__((__noinline__))
// CHECK: NOINLINE void macro_in_macro_with_attr() {}
NOINLINE void macro_in_macro_with_attr() {}
