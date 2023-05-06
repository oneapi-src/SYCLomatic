// RUN: dpct -in-root %S -out-root %T/header_order %S/test.cu %S/dnn.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/header_order/test.dp.cpp --match-full-lines %s
// RUN: FileCheck --input-file %T/header_order/dnn.dp.cpp --match-full-lines %s
// CHECK: #include <dpct/dnnl_utils.hpp>
// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
#include <stdlib.h>
#include <iostream>
#include <algorithm>

#include "dnn.h"

int main(){
  cudnnHandle_t handle;
  test(handle);
  return 0;
}
