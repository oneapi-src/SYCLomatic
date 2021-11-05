// RUN: dpct --format-range=none --usm-level=none -out-root %T/predefined_macro_replacement %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -D__NVCC__ -D__CUDACC__
// RUN: FileCheck --input-file %T/predefined_macro_replacement/predefined_macro_replacement.dp.cpp --match-full-lines %s
#include <stdio.h>
//CHECK: #ifdef DPCT_COMPATIBILITY_TEMP
//CHECK-NEXT: void hello(const sycl::stream &[[STREAM:stream_ct1]]) { [[STREAM]] << "foo"; }
#ifdef __CUDA_ARCH__
__global__ void hello() { printf("foo"); }
#else
__global__ void hello() { printf("other"); }
#endif

//CHECK: #ifndef DPCT_COMPATIBILITY_TEMP
#ifndef __NVCC__
__global__ void hello2() { printf("hello2"); }
#endif
//CHECK: #if defined(SYCL_LANGUAGE_VERSION)
#if defined(__CUDACC__)
__global__ void hello3() { printf("hello2"); }
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
//CHECK-NEXT: void hello5(const sycl::stream &[[STREAM]]) { [[STREAM]] << "hello2"; }
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
//CHECK-NEXT:     q_ct1.submit(
#if defined(__NVCC__)
  hello<<<1,1>>>();
#else
  hello();
#endif

//CHECK: #ifdef DPCT_COMPATIBILITY_TEMP
//CHECK-NEXT:     q_ct1.submit(
  #ifdef __NVCC__
  hello<<<1,1>>>();
#else
  hello();
#endif

//CHECK: #if DPCT_COMPATIBILITY_TEMP
//CHECK-NEXT:     q_ct1.submit(
  #if __NVCC__
  hello<<<1,1>>>();
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
//CHECK: #if SYCL_LANGUAGE_VERSION >= 4000
//CHECK-NEXT: dpct::get_current_device().reset();
//CHECK-NEXT: #else
//CHECK-NEXT: cudaThreadExit();
//CHECK-NEXT: #endif
#if CUDART_VERSION >= 4000
  cudaDeviceReset();
#else
  cudaThreadExit();
#endif
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
}