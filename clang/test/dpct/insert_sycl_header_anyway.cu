// RUN: dpct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/insert_sycl_header_anyway.dp.cpp --match-full-lines %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <stdio.h>
#include <stdio.h>
int main(){
  return 0;
}

