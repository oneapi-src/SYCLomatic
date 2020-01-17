// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/types004.dp.cpp

#include "cuda_fp16.h"

int main(int argc, char **argv) {
  //TODO: In CUDA 8.0 header file, __half and __half2 are defined by typedef an anonymous struct.
  //But in 9.2, 10.0 and 10.1, __half and __half2 are the names of struct.
  //Need refine the migration to cover this case.
  //CHECK:__half _h;
  //CHECK-NEXT:int a = sizeof(__half);
  //CHECK-NEXT:a = sizeof(_h);
  //CHECK-NEXT:a = sizeof _h;
  __half _h;
  int a = sizeof(__half);
  a = sizeof(_h);
  a = sizeof _h;

  //CHECK:__half2 _h2;
  //CHECK-NEXT:a = sizeof(__half2);
  //CHECK-NEXT:a = sizeof(_h2);
  //CHECK-NEXT:a = sizeof _h2;
  __half2 _h2;
  a = sizeof(__half2);
  a = sizeof(_h2);
  a = sizeof _h2;

}

