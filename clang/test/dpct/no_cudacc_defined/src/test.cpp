// RUN: cd %S/../build
// RUN: dpct -in-root ../src -out-root=%T -p ./  --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/test.cpp.dp.cpp
#include <iostream>
#include <cuda_runtime.h>
void test() {
  std::cout << "hello world" << std::endl;
}

// CHECK: #ifdef __CUDACC__
// CHECK-NEXT: template<typename T>
// CHECK-NEXT: __host__ __device__ T rounduptomult(T x, T m)
// CHECK-NEXT: {
// CHECK-NEXT:     return ((x + m - (T)1) / m) * m;
// CHECK-NEXT: }
// CHECK-NEXT: #endif
#ifdef __CUDACC__
template<typename T>
__host__ __device__ T rounduptomult(T x, T m)
{
    return ((x + m - (T)1) / m) * m;
}
#endif

// CHECK: #if __CUDACC__
// CHECK-NEXT: __global__ void k1(){
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
// CHECK-NEXT: #endif
#if __CUDACC__
__global__ void k1(){
  return 0;
}
#endif

// CHECK: #if defined(__CUDACC__)
// CHECK-NEXT: __global__ void k2(){
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
// CHECK-NEXT: #endif
#if defined(__CUDACC__)
__global__ void k2(){
  return 0;
}
#endif

int main() {
    test();
    return 0;
}
