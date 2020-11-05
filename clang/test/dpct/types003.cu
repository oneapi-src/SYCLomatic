// UNSUPPORTED: cuda-9.0, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v9.0, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/types003 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/types003/types003.dp.cpp


// __half  and __half2 has differnt definition between SDK 8.0 and >=9.2.
// In 8.0, it is a typedef of an anonymous struct, after 9.2 they are the names of struct.

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


