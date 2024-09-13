// RUN: dpct --out-root %T/ %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/a.cpp.dp.cpp --match-full-lines %s

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include "h1.h"
#include "h1.h"

int main() {
  return 0;
}
