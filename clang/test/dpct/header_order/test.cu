// RUN: dpct -in-root %S -out-root %T/header_order %S/test.cu %S/dnn.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/header_order/test.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/header_order/test.dp.cpp -o %T/header_order/test.dp.o %}
// RUN: FileCheck --input-file %T/header_order/dnn.dp.cpp --match-full-lines %S/dnn.cu
// RUN: %if build_lit %{icpx -c -fsycl %T/header_order/dnn.dp.cpp -o %T/header_order/dnn.dp.o %}
// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <stdlib.h>
// CHECK-NEXT: #include <iostream>
// CHECK-NEXT: #include <algorithm>
// CHECK-NEXT: #include "dnn.h"
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include "dnn.h"

int main() {
  cudnnHandle_t handle;
  test(handle);
  return 0;
}
