// RUN: dpct --format-range=none --usm-level=none -out-root %T/predefined_macro_replacement %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -D__NVCC__ -D__CUDACC__
// RUN: FileCheck --input-file %T/predefined_macro_replacement/predefined_macro_replacement.dp.cpp --match-full-lines %s
#include <stdio.h>
//CHECK: #ifdef DPCT_COMPATIBILITY_TEMP
//CHECK-NEXT: void hello(const sycl::stream &[[STREAM:stream_ct1]]) { [[STREAM]] << "foo"; }
#ifdef __CUDA_ARCH__
__device__ void hello() { printf("foo"); }
#else
__device__ void hello() { printf("other"); }
#endif

//CHECK: #ifndef DPCT_COMPATIBILITY_TEMP
#ifndef __NVCC__
__device__ void hello2() { printf("hello2"); }
#endif
//CHECK: #if defined(SYCL_LANGUAGE_VERSION)
#if defined(__CUDACC__)
__device__ void hello3() { printf("hello2"); }
#endif

#if defined(xxx)
__device__ void hello4() { printf("hello2"); }
//CHECK: #elif defined(DPCT_COMPATIBILITY_TEMP)
//CHECK-NEXT: void hello4(const sycl::stream &[[STREAM]]) { [[STREAM]] << "hello2"; }
#elif defined(__CUDA_ARCH__)
__device__ void hello4() { printf("hello2"); }
#endif

#if defined(xxx)
__device__ void hello5() { printf("hello2"); }
//CHECK: #elif (DPCT_COMPATIBILITY_TEMP >= 400)
//CHECK-NEXT: void hello5(const sycl::stream &[[STREAM]]) { [[STREAM]] << "hello2"; }
#elif (__CUDA_ARCH__ >= 400)
__device__ void hello5() { printf("hello2"); }
#endif

//CHECK: #if defined(DPCT_COMPATIBILITY_TEMP)
//CHECK-NEXT: void hello6(const sycl::stream &[[STREAM]]) { [[STREAM]] << "hello2"; }
#if defined(__CUDA_ARCH__)
__device__ void hello6() { printf("hello2"); }
#endif

//CHECK: #ifndef DPCT_COMPATIBILITY_TEMP
//CHECK-NEXT: __device__ void hello7() { printf("hello2"); }
//CHECK-NEXT: #else
//CHECK-NEXT: void hello7(const sycl::stream &[[STREAM]]) { [[STREAM]] << "hello2"; }
#ifndef __CUDA_ARCH__
__device__ void hello7() { printf("hello2"); }
#else
__device__ void hello7() { printf("hello2"); }
#endif

__global__ void hello8() {}

__device__ void test(){
//CHECK:#if (DPCT_COMPATIBILITY_TEMP >= 400) &&  (DPCT_COMPATIBILITY_TEMP >= 400)
//CHECK-NEXT:[[STREAM]] << ">400, \n";
//CHECK-NEXT:#elif (DPCT_COMPATIBILITY_TEMP >200)
//CHECK-NEXT:printf(">200, \n");
//CHECK-NEXT:#else
//CHECK-NEXT:printf("<200, \n");
//CHECK-NEXT:#endif
#if (__CUDA_ARCH__ >= 400) &&  (__CUDA_ARCH__ >= 400)
printf(">400, \n");
#elif (__CUDA_ARCH__ >200)
printf(">200, \n");
#else
printf("<200, \n");
#endif
}


int main() {
//CHECK: #if defined(DPCT_COMPATIBILITY_TEMP)
//CHECK-NEXT:     q_ct1.parallel_for(
#if defined(__NVCC__)
  hello8<<<1,1>>>();
#else
  hello();
#endif

//CHECK: #ifdef DPCT_COMPATIBILITY_TEMP
//CHECK-NEXT:     q_ct1.parallel_for(
  #ifdef __NVCC__
  hello8<<<1,1>>>();
#else
  hello();
#endif

//CHECK: #if DPCT_COMPATIBILITY_TEMP
//CHECK-NEXT:     q_ct1.parallel_for(
  #if __NVCC__
  hello8<<<1,1>>>();
#else
  hello();
#endif
  return 0;
}

//CHECK: #define AAA DPCT_COMPATIBILITY_TEMP
//CHECK-NEXT: #define BBB SYCL_LANGUAGE_VERSION
//CHECK-NEXT: #define CCC DPCT_COMPATIBILITY_TEMP
#define AAA __CUDA_ARCH__
#define BBB __CUDACC__
#define CCC __NVCC__

//CHECK: #ifdef __DPCT_HPP__
//CHECK-NEXT:#endif
//CHECK-NEXT:#ifdef __DPCT_HPP__
//CHECK-NEXT:#endif
#ifdef __DRIVER_TYPES_H__
#endif
#ifdef __CUDA_RUNTIME_H__
#endif

//CHECK: #if defined(__DPCT_HPP__)
//CHECK-NEXT:#endif
//CHECK-NEXT:#if defined(__DPCT_HPP__)
//CHECK-NEXT:#endif
#if defined(__DRIVER_TYPES_H__)
#endif
#if defined(__CUDA_RUNTIME_H__)
#endif

int foo(int num) {
//CHECK: #if (SYCL_LANGUAGE_VERSION >= 202000)
//CHECK-NEXT: dpct::get_current_device().reset();
//CHECK-NEXT: #else
//CHECK-NEXT: cudaThreadExit();
//CHECK-NEXT: #endif
#if CUDART_VERSION >= 4000
  cudaDeviceReset();
#else
  cudaThreadExit();
#endif

  return 0;

}

int foo1() {
//CHECK: #ifdef SYCL_LANGUAGE_VERSION
//CHECK-NEXT: sycl::int2 a;
//CHECK-NEXT: #endif
//CHECK-NEXT: #ifndef SYCL_LANGUAGE_VERSION
//CHECK-NEXT: int2 b;
//CHECK-NEXT: #endif
#ifdef CUDART_VERSION
int2 a;
#endif
#ifndef CUDART_VERSION
int2 b;
#endif
  a.x = 1;
  return a.x;
}

int foo2(){
  int version;
  //CHECK: int ret = (version = dpct::get_current_device().get_major_version(), 0);
  int ret = cudaRuntimeGetVersion(&version);
  int major = version / 1000;
  int minor = (version - major * 1000) / 10;
  int pl = version - major * 1000 - minor * 10;
  //CHECK: if (version != SYCL_LANGUAGE_VERSION) {
  //CHECK-NEXT:   major = SYCL_LANGUAGE_VERSION / 1000;
  //CHECK-NEXT:   minor = (SYCL_LANGUAGE_VERSION - major * 1000) / 10;
  //CHECK-NEXT:   pl = SYCL_LANGUAGE_VERSION - major * 1000 - minor * 10;
  //CHECK-NEXT: }
  if (version != CUDART_VERSION) {
    major = CUDART_VERSION / 1000;
    minor = (CUDART_VERSION - major * 1000) / 10;
    pl = CUDART_VERSION - major * 1000 - minor * 10;
  }
}

#define AAAA 1
//CHECK: void foo3() {
//CHECK-NEXT: #if defined SYCL_LANGUAGE_VERSION && (SYCL_LANGUAGE_VERSION >= 202000)
//CHECK-NEXT:   sycl::int2 a1;
//CHECK-NEXT: #endif
//CHECK-NEXT: #if defined(SYCL_LANGUAGE_VERSION) && (SYCL_LANGUAGE_VERSION >= 202000)
//CHECK-NEXT:   sycl::int2 a2;
//CHECK-NEXT: #endif
//CHECK-NEXT: #if (SYCL_LANGUAGE_VERSION >= 202000) && AAAA
//CHECK-NEXT:   sycl::int2 a3;
//CHECK-NEXT: #endif
//CHECK-NEXT: #if !(SYCL_LANGUAGE_VERSION >= 202000) && AAAA
//CHECK-NEXT:   int2 a4;
//CHECK-NEXT: #endif
//CHECK-NEXT: #if ((SYCL_LANGUAGE_VERSION >= 202000) ? 1 : 0) && AAAA
//CHECK-NEXT:   sycl::int2 a5;
//CHECK-NEXT: #endif
//CHECK-NEXT: }
void foo3() {
#if defined CUDART_VERSION && (CUDART_VERSION >= 4000)
  int2 a1;
#endif
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 4000)
  int2 a2;
#endif
#if (CUDART_VERSION >= 4000) && AAAA
  int2 a3;
#endif
#if !(CUDART_VERSION > 4000) && AAAA
  int2 a4;
#endif
#if (CUDART_VERSION > 4000 ? 1 : 0) && AAAA
  int2 a5;
#endif
}

//CHECK: void foo4() {
//CHECK-NEXT: #define BBBB 0
//CHECK-NEXT: #if BBBB
//CHECK-NEXT: #elif defined SYCL_LANGUAGE_VERSION && (SYCL_LANGUAGE_VERSION >= 202000)
//CHECK-NEXT:   sycl::int2 a1;
//CHECK-NEXT: #endif
//CHECK-NEXT: #if BBBB
//CHECK-NEXT: #elif defined(SYCL_LANGUAGE_VERSION) && (SYCL_LANGUAGE_VERSION >= 202000)
//CHECK-NEXT:   sycl::int2 a2;
//CHECK-NEXT: #endif
//CHECK-NEXT: #if BBBB
//CHECK-NEXT: #elif (SYCL_LANGUAGE_VERSION >= 202000) && AAAA
//CHECK-NEXT:   sycl::int2 a3;
//CHECK-NEXT: #endif
//CHECK-NEXT: #if BBBB
//CHECK-NEXT: #elif !(SYCL_LANGUAGE_VERSION >= 202000) && AAAA
//CHECK-NEXT:   int2 a4;
//CHECK-NEXT: #endif
//CHECK-NEXT: #if BBBB
//CHECK-NEXT: #elif ((SYCL_LANGUAGE_VERSION >= 202000) ? 1 : 0) && AAAA
//CHECK-NEXT:   sycl::int2 a5;
//CHECK-NEXT: #endif
//CHECK-NEXT: }
void foo4() {
#define BBBB 0
#if BBBB
#elif defined CUDART_VERSION && (CUDART_VERSION >= 4000)
  int2 a1;
#endif
#if BBBB
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 4000)
  int2 a2;
#endif
#if BBBB
#elif (CUDART_VERSION >= 4000) && AAAA
  int2 a3;
#endif
#if BBBB
#elif !(CUDART_VERSION > 4000) && AAAA
  int2 a4;
#endif
#if BBBB
#elif (CUDART_VERSION > 4000 ? 1 : 0) && AAAA
  int2 a5;
#endif
}
#undef BBBB

//CHECK: void foo5() {
//CHECK-NEXT: #define CCCC 1
//CHECK-NEXT: #if CCCC
//CHECK-NEXT: #elif defined SYCL_LANGUAGE_VERSION && (SYCL_LANGUAGE_VERSION >= 4000)
//CHECK-NEXT:   int2 a1;
//CHECK-NEXT: #endif
//CHECK-NEXT: #if CCCC
//CHECK-NEXT: #elif defined(SYCL_LANGUAGE_VERSION) && (SYCL_LANGUAGE_VERSION >= 4000)
//CHECK-NEXT:   int2 a2;
//CHECK-NEXT: #endif
//CHECK-NEXT: #if CCCC
//CHECK-NEXT: #elif (SYCL_LANGUAGE_VERSION >= 4000) && AAAA
//CHECK-NEXT:   int2 a3;
//CHECK-NEXT: #endif
//CHECK-NEXT: #if CCCC
//CHECK-NEXT: #elif !(SYCL_LANGUAGE_VERSION > 4000) && AAAA
//CHECK-NEXT:   int2 a4;
//CHECK-NEXT: #endif
//CHECK-NEXT: #if CCCC
//CHECK-NEXT: #elif (SYCL_LANGUAGE_VERSION > 4000 ? 1 : 0) && AAAA
//CHECK-NEXT:   int2 a5;
//CHECK-NEXT: #endif
//CHECK-NEXT: }
void foo5() {
#define CCCC 1
#if CCCC
#elif defined CUDART_VERSION && (CUDART_VERSION >= 4000)
  int2 a1;
#endif
#if CCCC
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 4000)
  int2 a2;
#endif
#if CCCC
#elif (CUDART_VERSION >= 4000) && AAAA
  int2 a3;
#endif
#if CCCC
#elif !(CUDART_VERSION > 4000) && AAAA
  int2 a4;
#endif
#if CCCC
#elif (CUDART_VERSION > 4000 ? 1 : 0) && AAAA
  int2 a5;
#endif
}
#undef CCCC
#undef AAAA
