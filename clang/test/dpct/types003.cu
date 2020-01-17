// UNSUPPORTED: cuda-9.0, cuda-9.2, cuda-10.0, cuda-10.1
// UNSUPPORTED: v9.0, v9.2, v10.0, v10.1
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/types003.dp.cpp

// __half  and __half2 has differnt define in cuda8 and cuda>=9.0.
// in cuda-8.0/include/cuda_fp16.h, line 94, it is a typedef of an anonymous struct:
// typedef struct __align__(2) {
//   unsigned short x;
// } __half;

// typedef struct __align__(4) {
//   unsigned int x;
// } __half2;

// But in 9.2, 10.0 and 10.1, __half and __half2 are the names of struct:
// cuda-10.0/include/cuda_fp16.h, line 126 and 135:
// struct __half;
// struct __half2;

#include "cuda_fp16.h"

int main(int argc, char **argv) {

  //CHECK:sycl::half _h;
  //CHECK-NEXT:int a = sizeof(sycl::half);
  //CHECK-NEXT:a = sizeof(_h);
  //CHECK-NEXT:a = sizeof _h;
  __half _h;
  int a = sizeof(__half);
  a = sizeof(_h);
  a = sizeof _h;

  //CHECK:sycl::half2 _h2;
  //CHECK-NEXT:a = sizeof(sycl::half2);
  //CHECK-NEXT:a = sizeof(_h2);
  //CHECK-NEXT:a = sizeof _h2;
  __half2 _h2;
  a = sizeof(__half2);
  a = sizeof(_h2);
  a = sizeof _h2;

}

