// RUN: dpct --format-range=none -out-root %T/warp_function_user_defined %s --cuda-include-path="%cuda-path/include" -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/warp_function_user_defined/warp_function_user_defined.dp.cpp

template<class T>
__device__ T __shfl_sync(unsigned int, T, T) {
  T a(1);
  return a;
}

//CHECK:void foo1() {
//CHECK-NEXT:  unsigned int mask = 1;
//CHECK-NEXT:  float a = 2;
//CHECK-NEXT:  float b = 3;
//CHECK-NEXT:  __shfl_sync(mask, a, b);
//CHECK-NEXT:}
__device__ void foo1() {
  unsigned int mask = 1;
  float a = 2;
  float b = 3;
  __shfl_sync(mask, a, b);
}
