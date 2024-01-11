// RUN: dpct --format-range=none --usm-level=none -in-root %S -out-root %T/cuda_arch_test_3 %S/test_3.cu -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_arch_test_3/test_3.dp.cpp
#include<cuda_runtime.h>
#include<iostream>

// CHECK: void test(){
// CHECK:   int a;
// CHECK:   unsigned long long b;
// CHECK:   a++;
// CHECK: }
// CHECK: void test_host_ct{{[0-9]+}}(){
// CHECK:   int a;
// CHECK:   unsigned long long b;
// CHECK:   b++;
// CHECK: }
// CHECK: void kernel(){
// CHECK:   test();
// CHECK: }
// CHECK: int main(){
// CHECK: #ifndef DPCT_COMPATIBILITY_TEMP 
// CHECK: test_host_ct{{[0-9]+}}();
// CHECK: #endif
// CHECK: }
__host__ __device__ void test(){

  int a;
  unsigned long long b;
#ifdef __CUDA_ARCH__
  a++;
#else
  b++;
#endif
}

__global__ void kernel(){

  test();
}


int main(){
#ifndef __CUDA_ARCH__ 
test();
#endif
}

