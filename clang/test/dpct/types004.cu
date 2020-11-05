// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/types004 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/types004/types004.dp.cpp

#include "cuda_fp16.h"

int main(int argc, char **argv) {
  //TODO: In SDK 8.0, __half and __half2 are defined by typedef an anonymous struct.
  //But after 9.2,they are the names of struct.
  //Need refine the migration to cover this case.
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


