// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -D__NVCC__  -D __CUDA_ARCH__ -D__CUDACC__
// RUN: FileCheck --input-file %T/predefined_macro_replacement.dp.cpp --match-full-lines %s
#include <stdio.h>
//CHECK: #ifdef DPCPP_COMPATIBILITY_TEMP
//CHECK-NEXT: /*
//CHECK-NEXT: DPCT1015:0: Adjust the code manually.
//CHECK-NEXT: */
//CHECK-NEXT: void hello(cl::sycl::stream [[STREAM:stream_ct1]]) { [[STREAM]] << "TODO - output needs update" << cl::sycl::endl; }
#ifdef __CUDA_ARCH__
__global__ void hello() { printf("intel"); }
#else
void hello() { printf("other"); }
#endif

//CHECK: #ifndef DPCPP_COMPATIBILITY_TEMP
#ifndef __NVCC__
void hello2() { printf("hello2"); }
#endif
//CHECK: #if defined(CL_SYCL_LANGUAGE_VERSION)
#if defined(__CUDACC__)
void hello3() { printf("hello2"); }
#endif

#if defined(xxx)
void hello4() { printf("hello2"); }
//CHECK: #elif defined(DPCPP_COMPATIBILITY_TEMP)
#elif defined(__CUDA_ARCH__)
void hello5() { printf("hello2"); }
#endif

__global__ void test(){
//CHECK:#if (DPCPP_COMPATIBILITY_TEMP >= 400) &&  (DPCPP_COMPATIBILITY_TEMP >= 400)
//CHECK-NEXT:printf(">400, \n");
//CHECK-NEXT:#elif (DPCPP_COMPATIBILITY_TEMP >200)
//CHECK-NEXT:printf(">200, \n");
//CHECK-NEXT:#else
//CHECK-NEXT: /*
//CHECK-NEXT: DPCT1015:1: Adjust the code manually.
//CHECK-NEXT: */
//CHECK-NEXT:[[STREAM]] << "TODO - output needs update" << cl::sycl::endl;
//CHECK-NEXT:#endif
#if (__CUDA_ARCH__ >= 400) &&  (__CUDA_ARCH__ >= 400)
printf(">400, \n");
#elif (__CUDA_ARCH__ >200)
printf(">200, \n");
#else
printf("<200 \n");
#endif
}


int main() {
//CHECK: #if defined(DPCPP_COMPATIBILITY_TEMP)
//CHECK-NEXT:     dpct::get_default_queue().submit(
#if defined(__NVCC__)
  hello<<<1,1>>>();
#else
  hello();
#endif
  return 0;
}
