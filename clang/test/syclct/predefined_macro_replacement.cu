// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path -D__NVCC__  -D __CUDA_ARCH__ -D__CUDACC__
// RUN: FileCheck --input-file %T/predefined_macro_replacement.sycl.cpp --match-full-lines %s
#include <stdio.h>
//CHECK: #ifdef DPCPP_COMPATIBILITY_TEMP
//CHECK-NEXT: void hello() { printf("intel"); }
#ifdef __CUDA_ARCH__
__global__ void hello() { printf("intel"); }
#else
void hello() { printf("other"); }
#endif

//CHECK: #ifndef DPCPP_COMPATIBILITY_TEMP
#ifndef __NVCC__
void hello2() { printf("hello2"); }
#endif
//CHECK: #if defined(DPCPP_COMPATIBILITY_TEMP)
#if defined(__CUDACC__)
void hello3() { printf("hello2"); }
#endif

#if defined(xxx)
void hello4() { printf("hello2"); }
//CHECK: #elif defined(DPCPP_COMPATIBILITY_TEMP)
#elif defined(__CUDA_ARCH__)
void hello5() { printf("hello2"); }
#endif
int main() {
//CHECK: #if defined(DPCPP_COMPATIBILITY_TEMP)
//CHECK-NEXT:   {
//CHECK-NEXT:     syclct::get_default_queue().submit(
#if defined(__NVCC__)
  hello<<<1,1>>>();
#else
  hello();
#endif
  return 0;
}
