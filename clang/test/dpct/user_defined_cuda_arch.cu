// RUN: dpct --format-range=none --usm-level=none -out-root %T/user_defined_cuda_arch %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -D__CUDA_ARCH__=200
// RUN: FileCheck --input-file %T/user_defined_cuda_arch/user_defined_cuda_arch.dp.cpp --match-full-lines %s
// RUN: dpct --format-range=none --usm-level=none -out-root %T/user_defined_cuda_arch %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -D__CUDA_ARCH__
// RUN: FileCheck --input-file %T/user_defined_cuda_arch/user_defined_cuda_arch.dp.cpp --match-full-lines %s
#include <stdio.h>
//CHECK: #ifdef DPCT_COMPATIBILITY_TEMP
//CHECK-NEXT: void hello(const sycl::stream &[[STREAM:stream_ct1]]) { [[STREAM]] << "foo"; }
#ifdef __CUDA_ARCH__
__global__ void hello() { printf("foo"); }
#else
__global__ void hello() { printf("other"); }
#endif

#if defined(xxx)
__global__ void hello4() { printf("hello2"); }
//CHECK: #elif defined(DPCT_COMPATIBILITY_TEMP)
//CHECK-NEXT: void hello4(const sycl::stream &[[STREAM]]) { [[STREAM]] << "hello2"; }
#elif defined(__CUDA_ARCH__)
__global__ void hello4() { printf("hello2"); }
#endif

#if defined(xxx)
__global__ void hello5() { printf("hello2"); }
//CHECK: #elif (DPCT_COMPATIBILITY_TEMP >= 400)
//CHECK-NEXT: __global__ void hello5() { printf("hello2"); }
#elif (__CUDA_ARCH__ >= 400)
__global__ void hello5() { printf("hello2"); }
#endif

//CHECK: #if defined(DPCT_COMPATIBILITY_TEMP)
//CHECK-NEXT: void hello6(const sycl::stream &[[STREAM]]) { [[STREAM]] << "hello2"; }
#if defined(__CUDA_ARCH__)
__global__ void hello6() { printf("hello2"); }
#endif

//CHECK: #ifndef DPCT_COMPATIBILITY_TEMP
//CHECK-NEXT: __global__ void hello7() { printf("hello2"); }
//CHECK-NEXT: #else
//CHECK-NEXT: void hello7(const sycl::stream &[[STREAM]]) { [[STREAM]] << "hello2"; }
#ifndef __CUDA_ARCH__
__global__ void hello7() { printf("hello2"); }
#else
__global__ void hello7() { printf("hello2"); }
#endif

__global__ void test(){
//CHECK:#if (DPCT_COMPATIBILITY_TEMP >= 400) &&  (DPCT_COMPATIBILITY_TEMP >= 400)
//CHECK-NEXT:printf(">400, \n");
//CHECK-NEXT:#elif (DPCT_COMPATIBILITY_TEMP >200)
//CHECK-NEXT:printf(">200, \n");
//CHECK-NEXT:#else
//CHECK-NEXT:[[STREAM]] << "<200, \n";
//CHECK-NEXT:#endif
#if (__CUDA_ARCH__ >= 400) &&  (__CUDA_ARCH__ >= 400)
printf(">400, \n");
#elif (__CUDA_ARCH__ >200)
printf(">200, \n");
#else
printf("<200, \n");
#endif
}
