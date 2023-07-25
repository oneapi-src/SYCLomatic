// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/half_raw %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/half_raw/half_raw.dp.cpp --match-full-lines %s

#include<iostream>
#include<cuda_runtime.h>
#include<cuda_fp16.h>
int main(){
  //CHECK: sycl::half one_h{sycl::bit_cast<sycl::half>(uint16_t(0x3C00))};
  __half_raw one_h{0x3C00};
  //CHECK: sycl::half zero_h{sycl::bit_cast<sycl::half>(uint16_t(0))};
  __half_raw zero_h{0};
  half alpha = one_h;
  std::cout<<(float)alpha<<'\n'; 
}